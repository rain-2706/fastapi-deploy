from fastapi import FastAPI, Query
import joblib
import uvicorn
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()
nb_open = open('nationality_predictor.pkl','rb')
model=joblib.load(nb_open)

cv = CountVectorizer()
@app.get('/predict/{name}')
async def nb_predict(name:str = Query(None, min_length=2, max_length=20)):
    vec = cv.transform([name]).toarray()
    result = model.predict(vec)

    return {"origin_name": name,'predict':result}


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost",port=8000, reload=True)