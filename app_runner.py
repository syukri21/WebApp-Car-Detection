from flask import Flask, session, g, render_template

app = Flask(__name__)

from home import home

app.register_blueprint(home.mod)
