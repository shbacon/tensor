import wavefile
import tensorflow as tf
import numpy as np
import math

def _fixed_frame(signal, frame_length, frame_step, first_axis=False):
    """tflite-compatible tf.signal.frame for fixed-size input.
    Args:
        signal: Tensor containing signal(s).
        frame_length: Number of samples to put in each frame.
        frame_step: Sample advance between successive frames.
        first_axis: If true, framing is applied to first axis of tensor; otherwise,
        it is applied to last axis.
    Returns:
        A new tensor where the last axis (or first, if first_axis) of input
        signal has been replaced by a (num_frames, frame_length) array of individual
        frames where each frame is drawn frame_step samples after the previous one.
    Raises:
        ValueError: if signal has an undefined axis length.  This routine only
        supports framing of signals whose shape is fixed at graph-build time.
    """
    signal_shape = signal.shape.as_list()
    if first_axis:
        length_samples = signal_shape[0]
    else:
        length_samples = signal_shape[-1]
    if length_samples <= 0:
        raise ValueError('fixed framing requires predefined constant signal length')
    num_frames = max(0, 1 + (length_samples - frame_length) // frame_step)
    if first_axis:
        inner_dimensions = signal_shape[1:]
        result_shape = [num_frames, frame_length] + inner_dimensions
        gather_axis = 0
    else:
        outer_dimensions = signal_shape[:-1]
        result_shape = outer_dimensions + [num_frames, frame_length]
        # Currently tflite's gather only supports axis==0, but that may still
        # work if we want the last of 1 axes.
        gather_axis = len(outer_dimensions)
    #subframe_length = fractions.gcd(frame_length, frame_step)  # pylint: disable=deprecated-method
    subframe_length = math.gcd(frame_length, frame_step)  # pylint: disable=deprecated-method
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = length_samples // subframe_length
    if first_axis:
        trimmed_input_size = [num_subframes * subframe_length] + inner_dimensions
        subframe_shape = [num_subframes, subframe_length] + inner_dimensions
    else:
        trimmed_input_size = outer_dimensions + [num_subframes * subframe_length]
        subframe_shape = outer_dimensions + [num_subframes, subframe_length]
    subframes = tf.reshape(
        tf.slice(
            signal,
            begin=np.zeros(len(signal_shape), np.int32),
            size=trimmed_input_size), subframe_shape)
    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = np.reshape(np.arange(num_frames) * subframes_per_hop, [num_frames, 1])
    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = np.reshape(np.arange(subframes_per_frame), [1, subframes_per_frame])
    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most subframes_per_frame
    # dimension to stitch the subframes together into frames. For example:
    # [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector
    frames = tf.reshape(tf.gather(subframes, selector.astype(np.int32), axis=gather_axis), result_shape)
    return frames

def _dft_matrix(dft_length):
  """Calculate the full DFT matrix in numpy."""
  omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
  # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
  return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

def _naive_rdft(signal_tensor, fft_length, padding='center'):
    """Implement real-input Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping
    # only the first half ("positive frequencies").
    # So discard the second half of rows, but transpose the array for
    # right-multiplication.
    # The DFT matrix is symmetric, so we could have done it more
    # directly, but this reflects our intention better.
    complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(fft_length // 2 + 1), :].transpose()
    real_dft_tensor = tf.constant(np.real(complex_dft_matrix_kept_values).astype(np.float32), name='real_dft_matrix')
    imag_dft_tensor = tf.constant(np.imag(complex_dft_matrix_kept_values).astype(np.float32), name='imaginary_dft_matrix')
    signal_frame_length = signal_tensor.shape[-1]#.value
    half_pad = (fft_length - signal_frame_length) // 2
    if padding == 'center':
        # Center-padding.
        pad_values = tf.concat([
            tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
            [[half_pad, fft_length - signal_frame_length - half_pad]]
        ], axis=0)
    elif padding == 'right':
        # Right-padding.
        pad_values = tf.concat([
            tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
            [[0, fft_length - signal_frame_length]]
        ], axis=0)
    padded_signal = tf.pad(signal_tensor, pad_values)
    result_real_part = tf.matmul(padded_signal, real_dft_tensor)
    result_imag_part = tf.matmul(padded_signal, imag_dft_tensor)
    return result_real_part, result_imag_part

def _stft_tflite(signal, frame_length, frame_step, fft_length):
    """tflite-compatible implementation of tf.signal.stft.
    Compute the short-time Fourier transform of a 1D input while avoiding tf ops
    that are not currently supported in tflite (Rfft, Range, SplitV).
    fft_length must be fixed. A Hann window is of frame_length is always
    applied.
    Since fixed (precomputed) framing must be used, signal.shape[-1] must be a
    specific value (so "?"/None is not supported).
    Args:
        signal: 1D tensor containing the time-domain waveform to be transformed.
        frame_length: int, the number of points in each Fourier frame.
        frame_step: int, the number of samples to advance between successive frames.
        fft_length: int, the size of the Fourier transform to apply.
    Returns:
        Two (num_frames, fft_length) tensors containing the real and imaginary parts
        of the short-time Fourier transform of the input signal.
    """
    # Make the window be shape (1, frame_length) instead of just frame_length
    # in an effort to help the tflite broadcast logic.
    window = tf.reshape(
        tf.constant(
            (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
            ).astype(np.float32),
            name='window'), [1, frame_length])
    framed_signal = _fixed_frame(signal, frame_length, frame_step, first_axis=False)
    framed_signal *= window
    real_spectrogram, imag_spectrogram = _naive_rdft(framed_signal, fft_length)
    return real_spectrogram, imag_spectrogram

#16K Sample, Mono, 1 Second -> Float32 Values
samples = wavefile.load('/Users/tbacon/Desktop/tensor/data/mini_speech_commands/yes/0c2d2ffa_nohash_0.wav')[1][0]

data = tf.constant(samples)
#spectrogram = tf.signal.stft(data, frame_length=255, frame_step=128)
#print(spectrogram)
spectrogram = _stft_tflite(data, 255, 128, 255) #Complex Number [0]+j[1]
print(spectrogram[0])
pad = tf.constant([[0, 0,], [0, 1]])
spectrogramPad = tf.pad(spectrogram[0], pad, "CONSTANT") #Added a missing element just padded with 0
spectrogram = spectrogramPad
spectrogram = tf.abs(spectrogram) #Calculates Magnitude of Complex Number
spectrogram = spectrogram[..., tf.newaxis]

interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.
interpreter.set_tensor(input['index'], [spectrogram])
interpreter.invoke()
print(interpreter.get_tensor(output['index']))
print(tf.nn.softmax(tf.constant(interpreter.get_tensor(output['index'])))) #Convert to Probabilities

#stft for lite - https://gist.github.com/padoremu/8288b47ce76e9530eb288d4eec2e0b4d
#fractions.gcd->math.gcd

#sftf for lite potential other option - https://kapre.readthedocs.io/en/latest/time_frequency_tflite.html
#spectrogram = kapre.time_frequency_tflite.STFTTflite(n_fft=255, hop_length=128)
