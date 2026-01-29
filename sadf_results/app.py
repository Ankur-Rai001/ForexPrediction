from flask import Flask, request, jsonify,url_for
from PredictMain import run_prediction_script, get_script_path

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data or 'currency' not in data or 'interval' not in data:
            return jsonify({'error': 'Missing currency or interval'}), 400
        currency = data['currency']
        interval = data['interval']
        imgurl=''
        if interval=='1day':
            imgurl=url_for('api/static', filename='gbpusd_forecast_1day.png', _external=True)
        if interval=='4hour':
            pass
        if interval=='1hour':
            pass
        if interval=='15min':
            pass


        # script_path = get_script_path(currency, interval)
        # result_data = run_prediction_script(script_path)
        if imgurl is None:
            app.logger.error(f"Failed to generate prediction data for {currency} {interval}")
            return jsonify({'error': 'Failed to generate prediction data'}), 500
        return jsonify({"imageurl":imgurl}), 200
    except ValueError as ve:
        app.logger.error(f"ValueError: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)