from flask import Flask  
from flask_restful import Resource, Api
from utils import say_hello

app = Flask(__name__)  
api = Api(app)

class Say_hello(Resource):  
    def get(self): 

        return say_hello()

api.add_resource(Say_hello, '/hello')

if __name__ == '__main__':  
    app.run(host='0.0.0.0', debug=False) 