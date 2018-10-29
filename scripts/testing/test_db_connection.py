from flask import Flask 
# from flaskext.mysqldb import MySQL 
from flaskext.mysql import MySQL
import pickle
from pprint import pprint



# app = Flask(__name__)
# app.config['MYSQL_HOST'] = '127.0.0.1'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'fashion_lens'
# mysql = MySQL(app)


# from flaskext.mysql import MySQL
# mysql = MySQL()
# # mysql = MySQL()
app = Flask(__name__)
mysql = MySQL()
# mySql Configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'fashion_lens'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'


# app.config['MYSQL_HOST'] = '127.0.0.1'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'fashion_lens'



mysql.init_app(app)

# conn = mysql.connection(

# cursor = conn.cursor()

@app.route('/')
def index():

	mydb = mysql.get_db()
	cur = mysql.get_db().cursor()

	# db_items = pickle.load( open('similar_item_pickle.p', 'rb'))

	sql = 'INSERT INTO `image_features` (`image_num`, `features`) VALUES (%s, %s);'
	val = pickle.load( open('ins_items.p', 'rb'))

	cur.executemany(sql, val)

	mydb.commit()

	# print(cur.rowcount, "was inserted.")

	# for key, values in db_items.items():
	# 	cur.execute(db_query)

if __name__ == '__main__':
	app.run(debug=True)
