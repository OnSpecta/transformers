from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime

args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8])

benchmark = TensorFlowBenchmark(args)

# Set up logging.
# stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = '/onspecta/dev/logs/transformers_logs/%s' % stamp
# writer = tf.summary.create_file_writer(logdir)

# tf.summary.trace_on(graph=True, profiler=True)
results = benchmark.run()
# with writer.as_default():
#     tf.summary.trace_export(
#         name="my_func_trace",
#         step=0,
#         profiler_outdir=logdir)

print(results)

