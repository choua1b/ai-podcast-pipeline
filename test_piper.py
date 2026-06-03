from piper import PiperVoice
import inspect

sig1 = inspect.signature(PiperVoice.synthesize_wav)
print("synthesize_wav signature:", sig1)

sig2 = inspect.signature(PiperVoice.synthesize)
print("synthesize signature:", sig2)

voice = PiperVoice.load(
    'voices/en_US-lessac-medium.onnx',
    config_path='voices/en_US-lessac-medium.onnx.json',
    use_cuda=False
)
print("Model loaded OK")
print("Sample rate:", voice.config.sample_rate)

result = voice.synthesize("Hello world, this is a test.")
print("synthesize() returned type:", type(result))
print("synthesize() value:", result)