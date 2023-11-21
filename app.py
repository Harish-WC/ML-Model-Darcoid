import model_handler as wcfcmodel
import json
import os
import flask

app = flask.Flask(__name__)

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


@app.route("/ping", methods=["GET"])
def ping():
    status = 200
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def projections():
    """Do an inference on a single batch of data. In this sample server, we take data as JSON, 
    and then pass it to the wcfcmodel.handle function.
    """
    print("Invoking the model...")
    data = json.loads(request.data)
    result = wcfcmodel.handle(data, "")
    print(result)
    print("Done Invocation!")
    return result
