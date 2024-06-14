import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

app = Flask(__name__)
CORS(app)

df = pd.read_csv('/Users/nanthansnair/Downloads/jupiter/data1.csv')
df['Actual flow volume air/gas'] = df['Actual flow volume air/gas'].str.replace(',', '').str.replace(' m3/h', '').astype(float)
df['Pressure, static'] = df['Pressure, static'].str.replace(',', '').str.replace(' Pa', '').astype(float)
df['Rated power'] = df['Rated power'].str.replace(' kW', '').astype(float)
df.drop(columns=['Unnamed: 5'], inplace=True)
df.dropna(inplace=True)

X = df[['Actual flow volume air/gas', 'Pressure, static']]
y = df['Rated power']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

def recommend_fan(flow, pressure):
    try:
        input_scaled = scaler.transform(np.array([[flow, pressure]]))
        predicted_power = model.predict(input_scaled)[0][0]
        df['Power Difference'] = abs(df['Rated power'] - predicted_power)
        sorted_fans = df.sort_values(by='Power Difference', ascending=True)
        best_fan = sorted_fans.iloc[0]
        return best_fan.to_dict()
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Tool</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 0;
          padding: 0;
          background-color: #f7f7f7; 
        }
        header {
          background-color: #3f51b5; 
          padding: 10px;
          color: #fff; 
          text-align: center; 
          height: 55px;
          position: fixed;
          width: 100%;
          top: 0;
          z-index: 1000;
        }
        h1 {
          margin-top: -45px;
        }
        nav a {
          text-decoration: none;
          margin-right: 10px;
          color: #fff; 
          transition: color 0.3s ease; 
        }
        nav a:hover {
          color: #ccc; 
        }
        main {
          padding: 20px;
        }
        section {
          margin-bottom: 20px;
          text-align: center; 
        }
        .image-gallery {
          display: grid;
          grid-template-columns: repeat(3, 1fr); 
          gap: 20px;
          justify-items: center; 
        }
        .image-container {
          display: flex;
          justify-content: center; 
        }
        .image-container img {
          max-width: 600px; 
          max-height: 500px; 
        }
        footer {
          background-color: #3f51b5; 
          color: #fff; 
          padding: 10px;
          position:static;
          left: 0;
          bottom: 0;
          width: 100%;
          text-align: center;
        }
        .container {
          padding: 20px;
          margin-top: 60px;
          width: 1000px;
          text-align: center;
        }
        .result-container {
          border: 2px solid black;
          padding: 20px;
          margin-top: 20px;
          width: 1490px;
          text-align: center;
        }
        .image-container {
          display: flex;
          justify-content: center;
          gap: 10px;
        }
        .image-container img {
          height: 500px;
          border: 2px solid black;
        }
        .logo {
          height: 50px;
          width: 150px;
          margin-left: -1310px;
        }
        .form1{
          display: grid;
          grid-template-columns: 1fr 1fr;
          row-gap: 19px;
          column-gap: 15px;
          justify-items: center;
        }
        form div {
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        form label {
          margin-bottom: 5px;
        }
        button {
          grid-column: span 2;
          border-radius: 14px;
          padding: 10px 20px;
          margin-top: 10px;
        }
        @media screen and (max-width: 600px) {
          .container {
            width: 90%; 
          }
          .result-container {
            width: 90%; 
          }
          .image-container img {
            height: auto; 
            max-width: 100%; 
          }
          form {
            grid-template-columns: 1fr;
          }
          button {
            grid-column: span 1;
          }
        }
      </style>
    <script>
      async function recommendFan(event) {
          event.preventDefault();
          const flow = document.getElementById("flow").value;
          const pressure = document.getElementById("pressure").value;
          try {
              const response = await fetch('http://127.0.0.1:5001/recommend_fan', {
                  method: 'POST',
                  headers: {
                      'Content-Type': 'application/json'
                  },
                  body: JSON.stringify({ flow: flow, pressure: pressure })
              });
              if (response.ok) {
                  const data = await response.json();
                  const resultWindow = window.open("", "_blank");
                  resultWindow.document.write("<div style='height: 96%; width: 80%; margin: 0 auto; border: 2px solid black;'>");
                    resultWindow.document.write("<header style='background-color: #3f51b5;color: #fff;text-align: center; height: 120px;'>")
                    resultWindow.document.write("<img src='static/logo.png' alt='footerlogo' style='height: 50px;width: 150px;'>")
                    resultWindow.document.write("<h1 style='text-align: center;'>Recommended System Of Fan</h1>");
                    resultWindow.document.write("</header>")
                    resultWindow.document.write("<div style='padding-left: 15px; text-align: left;'>");
                    resultWindow.document.write(formatData(data));
                    resultWindow.document.write("</div>")
                    resultWindow.document.write("<div style='display: flex; justify-content: center; gap: 10px; margin-top: 20px;'>");
                    resultWindow.document.write("<img src='static/img3.jpeg' alt='gp1' style='height: 400px;'>")
                    resultWindow.document.write("<img src='static/img4.jpeg' alt='gp1' style='height: 400px;'>")
                    resultWindow.document.write("</div>")
                    resultWindow.document.write("<center>")
                    resultWindow.document.write("<button style='height: 23px; width: 80px; border-radius: 14px;'>")
                    resultWindow.document.write("<a href='static/Variation_Report.pdf' download='Report'>Report</a>")
                    resultWindow.document.write("</button>")
                    resultWindow.document.write("</center>")
                    resultWindow.document.write("</div>")
              } else {
                  alert('Error: ' + response.statusText);
              }
          } catch (error) {
              console.error('Error:', error);
              alert('error');
          }
      }
      function formatData(data) {
          let formattedData = "<div style='font-family: Arial, sans-serif;'>";
          for (const [key, value] of Object.entries(data)) {
              formattedData += `<p><strong>${key}:</strong> ${value}</p>`;
          }
          formattedData += "</div>";
          return formattedData;
      }
    </script>
    </head>
    <body>
      <header>
        <img src="static/logo.png" alt="footerlogo" class="logo">
        <h1>Enter The Values of  Parameters</h1>
      </header>

      <main>
        <section>
          <center>
            <div class="container">
              <form onsubmit="recommendFan(event)">
                <label for="flow">Flow (m3/h): </label>
                <input type="number" id="flow" name="flow" required>
                <br><br>
                <label for="pressure">Pressure (Pa): </label>
                <input type="number" id="pressure" name="pressure" required>
                <br><br>
                <button type="submit">Recommend Fan</button>
              </form>
              <br>
              <h2>Optional Parameters</h2>
              <br>
              <form class="form1">
                <div>
                  <label for="optional1">Type </label>
                  <input type="text" id="optional1" name="optional1">
                </div>
                <div>
                  <label for="optional2">Op. temp. (°C)</label>
                  <input type="number" id="optional2" name="optional2">
                </div>
                <div>
                  <label for="optional3">VFD </label>
                  <input type="text" id="optional3" name="optional3">
                </div>
                <div>
                  <label for="optional4">Type of drive </label>
                  <input type="text" id="optional4" name="optional4">
                </div>
                <div>
                  <label for="optional5">Outlet orientation / Design </label>
                  <input type="text" id="optional7" name="optional7">
                </div>
                <div>
                  <label for="optional6">Material specification </label>
                  <input type="text" id="optional8" name="optional8">
                </div>
                <div>
                  <label for="optional7">Material impeller</label>
                  <input type="text" id="optional9" name="optional9">
                </div>
                <div>
                  <label for="optional18">Effeciency class</label>
                  <input type="text" id="optional10" name="optional10">
                </div>
              </form>
            </div>
            <br>
            <div class="image-container">
              <img src="static/img1.jpeg" alt="Image 1">
              <img src="static/img2.jpeg" alt="Image 2">
            </div>
          </center>
        </section>
      </main>
      <footer>
        <img src="static/footer1.png" alt="footerlogo" class="logo">
      </footer>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/recommend_fan', methods=['POST'])
def recommend_fan_endpoint():
    try:
        data = request.get_json()
        flow = data['flow']
        pressure = data['pressure']
        recommended_fan = recommend_fan(flow, pressure)
        return jsonify(recommended_fan)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
