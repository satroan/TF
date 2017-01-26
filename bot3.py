import tensorflow as tf
import data_utils
import seqseq_model


def train():
	#Dataset
	enc_train, dec_train = data_utils.prepare_custom_data(gConfig['working directory'])
	train_set = read_data(enc_train, dec_train)
def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
	return tf.nn.seq2seq.embedding_attention_seq2seq(
		encoder_inputs, decoder_inputs, cell,
		num_encoder_symbols=source_vocab_size,
		num_decoder_symbols=target_vocab_size,
		embedding_size=size,
		output_projection=output projection,
		feed_previous=do_decode)
with tf.Session(config=config) as sess:
	#Create Model
	model = create_model(sess, False)

	while True:
		sess.run(model)
		#Save checkpoint and zero timer and losses as well as other stuff down the line
		checkpoint_path = os.path.join(gConfig['working directory'], "seq2seq.ckpt")
		model.saver.save(sess, checkpoint_path, global_step=model.global_step)