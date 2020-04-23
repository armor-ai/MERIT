import engine
import time


timed_reviews = engine.extract_review()
OLDA_input = engine.build_input_version(timed_reviews)
start_t = time.time()
apk_phis, topic_dict = engine.OLDA_fit(OLDA_input)
phrases = engine.generate_labeling_candidates(OLDA_input)
engine.topic_labeling_n(OLDA_input, apk_phis, phrases, topic_dict)
print("Totally takes %.2f seconds" % (time.time() - start_t))
