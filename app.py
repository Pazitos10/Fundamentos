from flask import Flask, render_template, request
from classifiersTest import process_data

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.form['matriz_canvas']
    predictions, probs = process_data(data)
    return render_template('results.html', 
    	img_path="static/img/img.png", 
    	predictions=predictions, 
    	probs=probs)


    

if __name__ == "__main__":
    app.run(debug=True)