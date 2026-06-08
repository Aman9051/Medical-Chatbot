from flask import Flask, render_template, request
from src.rag_chain import load_chain

app = Flask(__name__)

chain = load_chain()

@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""
    question = ""

    if request.method == "POST":
        question = request.form["question"]
        answer = chain.invoke(question)

    return render_template(
        "chat.html",
        question=question,
        answer=answer
    )


if __name__ == "__main__":
    app.run(debug=True)