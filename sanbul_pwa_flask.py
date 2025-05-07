import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.preprocessing import StandardScaler

model = keras.models.load_model("fires_model.keras")

app = Flask(__name__)
app.config["SECRET_KEY"] = "hard to guess string"
bootstrap5 = Bootstrap5(app)

num_attribs = ["longitude","latitude","avg_temp","max_temp","max_wind_speed","avg_wind"]

class LabForm(FlaskForm):
    longitude      = StringField('Longitude (1~7)',           validators=[DataRequired()])
    latitude       = StringField('Latitude (1~7)',            validators=[DataRequired()])
    month          = StringField('Month (01-Jan ~ 12-Dec)',   validators=[DataRequired()])
    day            = StringField('Day (00-sun ~ 07-hol)',      validators=[DataRequired()])
    avg_temp       = StringField('Avg Temp',                  validators=[DataRequired()])
    max_temp       = StringField('Max Temp',                  validators=[DataRequired()])
    max_wind_speed = StringField('Max Wind Speed',            validators=[DataRequired()])
    avg_wind       = StringField('Avg Wind',                  validators=[DataRequired()])
    submit         = SubmitField('Predict')

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["GET","POST"])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        data = {
            "longitude":      [float(form.longitude.data)],
            "latitude":       [float(form.latitude.data)],
            "month":          [form.month.data],   # 지금은 모델에 사용하지 않음
            "day":            [form.day.data],     # 지금은 모델에 사용하지 않음
            "avg_temp":       [float(form.avg_temp.data)],
            "max_temp":       [float(form.max_temp.data)],
            "max_wind_speed": [float(form.max_wind_speed.data)],
            "avg_wind":       [float(form.avg_wind.data)],
        }
        df_input = pd.DataFrame(data)
        
        # 전체 학습 데이터로 스케일러 재생성 (배포 환경에서는 미리 저장해 두는 게 바람직)
        df_all = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
        df_all["burned_area"] = np.log(df_all["burned_area"] + 1)
        scaler = StandardScaler()
        scaler.fit(df_all[num_attribs])
        
        # 숫자형 특성만 스케일링
        X_scaled = scaler.transform(df_input[num_attribs])
        y_pred = model.predict(X_scaled)
        burned_area = float(y_pred[0][0])
        
        return render_template("result.html", burned_area=burned_area)
    return render_template("prediction.html", form=form)

if __name__ == "__main__":
    app.run(debug=True)
