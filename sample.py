from bottle import route, run, HTTPResponse, template
import os

@route("/sample.json")
def json():
    body = {"status": "OK", "message": "hello world"}
    r = HTTPResponse(status=200, body=body)
    r.set_header("Content-Type", "application/json")
    return template("free")

run(host="localhost", port=8080)
