from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('F:/UNAIR/ALGORITMA PEMROGRAMAN II/Final Project/FIX BGT/prediksi_pinjem.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to one-hot encode home_ownership and loan_intent
def one_hot_encode(person_home_ownership, loan_intent):
    home_ownership_map = {'OWN': [1, 0, 0], 'RENT': [0, 1, 0], 'OTHER': [0, 0, 1]}
    loan_intent_map = {
        'EDUCATION': [1, 0, 0, 0, 0],
        'HOMEIMPROVEMENT': [0, 1, 0, 0, 0],
        'MEDICAL': [0, 0, 1, 0, 0],
        'PERSONAL': [0, 0, 0, 1, 0],
        'VENTURE': [0, 0, 0, 0, 1]
    }

    home_ownership_encoded = home_ownership_map.get(person_home_ownership, [0, 0, 0])
    loan_intent_encoded = loan_intent_map.get(loan_intent, [0, 0, 0, 0, 0])

    return home_ownership_encoded + loan_intent_encoded

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman simulasi
@app.route('/simulasi')
def simulasi():
    return render_template('simulasi.html')

# Route untuk proses prediksi
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    person_age = int(data['person_age'])
    person_income = float(data['person_income'])
    person_home_ownership = data['person_home_ownership']
    person_emp_length = float(data['person_emp_length'])
    loan_intent = data['loan_intent']
    loan_amnt = float(data['loan_amnt'])
    
    # Calculate loan_percent_income
    loan_percent_income = (loan_amnt / person_income) * 100

    # One-hot encode features
    encoded_features = one_hot_encode(person_home_ownership, loan_intent)

    # Combine input data
    input_data = [
        person_age,
        person_income,
        person_emp_length,
        loan_amnt,
        loan_percent_income,
    ] + encoded_features

    # Predict using the model
    result = model.predict([input_data])[0]

    # Build the response
    prediction = "Layak Pinjam" if result == 0 else "Tidak Layak Pinjam"
    reason = f"Jumlah pinjaman Anda adalah {loan_amnt}, dengan pendapatan tahunan {person_income}."
    suggestion = "Jaga stabilitas keuangan Anda." if result == 0 else "Pertimbangkan untuk mengurangi jumlah pinjaman."

    response = {
        'prediction': prediction,
        'reason': reason,
        'suggestion': suggestion,
        'input_data': data
    }

    return jsonify(response)

@app.route('/analisis')
def analisis():
    return render_template('analisis.html')

@app.route('/calculate', methods=['POST'])
def calculate_cicilan():
    data = request.json
    loan_amount = float(data['loanAmount'])  # Jumlah pinjaman
    loan_tenor = int(data['loanTenor'])      # Lama pinjaman dalam bulan
    loan_provider = data['loanProvider']     # Layanan kredit yang dipilih

    # Bunga per tahun berdasarkan layanan kredit
    interest_rates = {
        "Gopaylater": 0.41,    # 41%
        "Shopee": 0.48,        # 48%
        "Uangme": 0.39,        # 39%
        "KreditPintar": 0.44,  # 44%
        "Julo": 0.24           # 24%
    }

    # Ambil bunga tahunan; jika tidak ditemukan, default ke 40%
    annual_interest = interest_rates.get(loan_provider, 0.4)
    monthly_interest_rate = annual_interest / 12  # Bunga per bulan

    # Aturan diskon: jika tenor <= 6 bulan, kurangi bunga 1%
    if loan_tenor <= 6:
        monthly_interest_rate -= 0.01

    # Hitung total pembayaran dan cicilan per bulan
    total_payment = loan_amount * (1 + monthly_interest_rate * loan_tenor)
    cicilan_per_bulan = total_payment / loan_tenor

    # Response JSON
    response = {
        'cicilan': round(cicilan_per_bulan, 2),
        'total_payment': round(total_payment, 2),
        'monthly_interest_rate': round(monthly_interest_rate * 100, 2)  # Dalam persen
    }
    return jsonify(response)

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
