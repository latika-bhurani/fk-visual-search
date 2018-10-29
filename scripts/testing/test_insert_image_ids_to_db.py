import pickle
from pprint import pprint

db_items = pickle.load( open('similar_item_pickle.p', 'rb'))

ins_item = []
for key, value in db_items.items():
# db = db_items.items()
	ins_item.append((int(key), str(value)))


pickle.dump(ins_item, open('ins_items.p', 'wb'))
print(ins_item)
print(len(ins_item))