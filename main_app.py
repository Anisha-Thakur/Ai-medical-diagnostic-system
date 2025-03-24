import streamlit as st
import pickle
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from streamlit_option_menu import option_menu

# Change Name & Logo
st.set_page_config(page_title="AI Medical Diagnosis System", page_icon="⚕️")

# Hiding Streamlit add-ons
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Adding Background Image
background_image_url = "https://i.ibb.co/9mHbLgm8/bg.jpg"  # Replace with your image URL

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.7);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the saved models
models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
}

# Initialize speech recognizer
recognizer = sr.Recognizer()

def text_to_speech(text):
    """Convert text to speech and play it"""
    tts = gTTS(text=text, lang='en')
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    st.audio(audio_file, format='audio/mp3')

def get_voice_input():
    """Get input from microphone"""
    try:
        with sr.Microphone() as source:
            st.session_state.listening = True
            st.write("Listening... Speak now")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.session_state.listening = False
            return text
    except sr.WaitTimeoutError:
        st.warning("Listening timed out. Please try again.")
        return None
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    finally:
        st.session_state.listening = False

def display_input(label, tooltip, key, type="text"):
    """Display input field with voice input option and audio tooltip"""
    col1, col2, col3 = st.columns([4, 1, 1])
    
    # Initialize session state for this key if not exists
    if key not in st.session_state:
        st.session_state[key] = ""
    
    with col1:
        if type == "text":
            input_value = st.text_input(label, value=st.session_state[key], key=f"input_{key}")
        elif type == "number":
            try:
                num_value = float(st.session_state[key]) if st.session_state[key] else 0.0
                input_value = st.number_input(label, value=num_value, key=f"input_{key}", step=1.0)
            except:
                input_value = st.number_input(label, value=0.0, key=f"input_{key}", step=1.0)
    
    with col2:
        if st.button("🎤", key=f"mic_{key}", disabled=st.session_state.get("listening", False)):
            voice_input = get_voice_input()
            if voice_input:
                st.session_state[key] = voice_input
                st.rerun()
    
    with col3:
        if st.button("ℹ️", key=f"info_{key}"):
            text_to_speech(tooltip)
    
    # Update the widget value if changed via voice
    return st.session_state[key]

# Home Page Navigation
if 'selected_disease' not in st.session_state:
    # Home Page Layout
    st.title("Welcome to AI Medical Diagnosis System")
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 18px;'>
            This system helps predict various diseases using machine learning models. 
            Select a disease from the dropdown menu below to begin diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected = st.selectbox(
            'Select a Disease to Predict',
            ['Diabetes Prediction',
             'Heart Disease Prediction',
             'Parkinsons Prediction',
             'Lung Cancer Prediction',
             'Hypo-Thyroid Prediction'],
            key='home_select'
        )
        
        if st.button("Start Diagnosis", key="start_diagnosis"):
            st.session_state.selected_disease = selected
            st.rerun()
else:
    # Back Button
    if st.button("← Back to Home"):
        del st.session_state.selected_disease
        st.rerun()
    
    # Disease Pages
    selected = st.session_state.selected_disease
    
    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':
        st.title('Diabetes Prediction')
        st.write("Enter the following details to predict diabetes:")

        Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies', 'number')
        Glucose = display_input('Glucose Level', 'Enter glucose level in mg/dL', 'Glucose', 'number')
        BloodPressure = display_input('Blood Pressure value', 'Enter blood pressure value in mmHg', 'BloodPressure', 'number')
        SkinThickness = display_input('Skin Thickness value', 'Enter skin thickness value at triceps in mm', 'SkinThickness', 'number')
        Insulin = display_input('Insulin Level', 'Enter insulin level in mu U/mL', 'Insulin', 'number')
        BMI = display_input('BMI value', 'Enter Body Mass Index value in kg/m²', 'BMI', 'number')
        DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function value', 'Enter diabetes pedigree function value', 'DiabetesPedigreeFunction', 'number')
        Age = display_input('Age of the Person', 'Enter age of the person in years', 'Age', 'number')

        if st.button('Diabetes Test Result'):
            try:
                input_values = [
                    float(Pregnancies), float(Glucose), float(BloodPressure),
                    float(SkinThickness), float(Insulin), float(BMI),
                    float(DiabetesPedigreeFunction), float(Age)
                ]
                diab_prediction = models['diabetes'].predict([input_values])
                diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
                st.success(diab_diagnosis)
                text_to_speech(diab_diagnosis)
            except ValueError:
                st.error("Please enter valid numbers for all fields")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # Heart Disease Prediction Page
    elif selected == 'Heart Disease Prediction':
        st.title('Heart Disease Prediction')
        st.write("Enter the following details to predict heart disease:")

        age = display_input('Age', 'Enter age of the person in years', 'age', 'number')
        sex = display_input('Sex (1 = male; 0 = female)', 'Enter 1 for male or 0 for female', 'sex', 'number')
        cp = display_input('Chest Pain types (0, 1, 2, 3)', 'Enter chest pain type: 0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, 3 for asymptomatic', 'cp', 'number')
        trestbps = display_input('Resting Blood Pressure', 'Enter resting blood pressure in mmHg', 'trestbps', 'number')
        chol = display_input('Serum Cholesterol in mg/dl', 'Enter serum cholesterol in mg/dL', 'chol', 'number')
        fbs = display_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', 'Enter 1 if fasting blood sugar is greater than 120 mg/dL, otherwise 0', 'fbs', 'number')
        restecg = display_input('Resting Electrocardiographic results (0, 1, 2)', 'Enter resting ECG results: 0 for normal, 1 for ST-T wave abnormality, 2 for probable or definite left ventricular hypertrophy', 'restecg', 'number')
        thalach = display_input('Maximum Heart Rate achieved', 'Enter maximum heart rate achieved', 'thalach', 'number')
        exang = display_input('Exercise Induced Angina (1 = yes; 0 = no)', 'Enter 1 if exercise induced angina is present, otherwise 0', 'exang', 'number')
        oldpeak = display_input('ST depression induced by exercise', 'Enter ST depression induced by exercise relative to rest', 'oldpeak', 'number')
        slope = display_input('Slope of the peak exercise ST segment (0, 1, 2)', 'Enter slope of the peak exercise ST segment: 0 for upsloping, 1 for flat, 2 for downsloping', 'slope', 'number')
        ca = display_input('Major vessels colored by fluoroscopy (0-3)', 'Enter number of major vessels colored by fluoroscopy (0-3)', 'ca', 'number')
        thal = display_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)', 'Enter thalassemia type: 0 for normal, 1 for fixed defect, 2 for reversible defect', 'thal', 'number')

        if st.button('Heart Disease Test Result'):
            try:
                input_values = [
                    float(age), float(sex), float(cp), float(trestbps),
                    float(chol), float(fbs), float(restecg), float(thalach),
                    float(exang), float(oldpeak), float(slope), float(ca), float(thal)
                ]
                heart_prediction = models['heart_disease'].predict([input_values])
                heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
                st.success(heart_diagnosis)
                text_to_speech(heart_diagnosis)
            except ValueError:
                st.error("Please enter valid numbers for all fields")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # Parkinson's Prediction Page
    elif selected == "Parkinsons Prediction":
        st.title("Parkinson's Disease Prediction")
        st.write("Enter the following details to predict Parkinson's disease:")

        fo = display_input('MDVP:Fo(Hz)', 'Enter average vocal fundamental frequency in Hz', 'fo', 'number')
        fhi = display_input('MDVP:Fhi(Hz)', 'Enter maximum vocal fundamental frequency in Hz', 'fhi', 'number')
        flo = display_input('MDVP:Flo(Hz)', 'Enter minimum vocal fundamental frequency in Hz', 'flo', 'number')
        Jitter_percent = display_input('MDVP:Jitter(%)', 'Enter jitter percentage', 'Jitter_percent', 'number')
        Jitter_Abs = display_input('MDVP:Jitter(Abs)', 'Enter absolute jitter in microseconds', 'Jitter_Abs', 'number')
        RAP = display_input('MDVP:RAP', 'Enter relative amplitude perturbation', 'RAP', 'number')
        PPQ = display_input('MDVP:PPQ', 'Enter five-point period perturbation quotient', 'PPQ', 'number')
        DDP = display_input('Jitter:DDP', 'Enter average absolute difference of differences between consecutive periods', 'DDP', 'number')
        Shimmer = display_input('MDVP:Shimmer', 'Enter shimmer', 'Shimmer', 'number')
        Shimmer_dB = display_input('MDVP:Shimmer(dB)', 'Enter shimmer in decibels', 'Shimmer_dB', 'number')
        APQ3 = display_input('Shimmer:APQ3', 'Enter three-point amplitude perturbation quotient', 'APQ3', 'number')
        APQ5 = display_input('Shimmer:APQ5', 'Enter five-point amplitude perturbation quotient', 'APQ5', 'number')
        APQ = display_input('MDVP:APQ', 'Enter amplitude perturbation quotient', 'APQ', 'number')
        DDA = display_input('Shimmer:DDA', 'Enter average absolute difference between consecutive differences between amplitudes', 'DDA', 'number')
        NHR = display_input('NHR', 'Enter noise-to-harmonics ratio', 'NHR', 'number')
        HNR = display_input('HNR', 'Enter harmonics-to-noise ratio', 'HNR', 'number')
        RPDE = display_input('RPDE', 'Enter recurrence period density entropy', 'RPDE', 'number')
        DFA = display_input('DFA', 'Enter detrended fluctuation analysis', 'DFA', 'number')
        spread1 = display_input('Spread1', 'Enter nonlinear measure of fundamental frequency variation', 'spread1', 'number')
        spread2 = display_input('Spread2', 'Enter nonlinear measure of fundamental frequency variation', 'spread2', 'number')
        D2 = display_input('D2', 'Enter nonlinear dynamical complexity measure', 'D2', 'number')
        PPE = display_input('PPE', 'Enter pitch period entropy', 'PPE', 'number')

        if st.button("Parkinson's Test Result"):
            try:
                input_values = [
                    float(fo), float(fhi), float(flo), float(Jitter_percent),
                    float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                    float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                    float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                    float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
                ]
                parkinsons_prediction = models['parkinsons'].predict([input_values])
                parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
                st.success(parkinsons_diagnosis)
                text_to_speech(parkinsons_diagnosis)
            except ValueError:
                st.error("Please enter valid numbers for all fields")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # Lung Cancer Prediction Page
    elif selected == "Lung Cancer Prediction":
        st.title("Lung Cancer Prediction")
        st.write("Enter the following details to predict lung cancer:")

        GENDER = display_input('Gender (1 = Male; 0 = Female)', 'Enter 1 for male or 0 for female', 'GENDER', 'number')
        AGE = display_input('Age', 'Enter age of the person in years', 'AGE', 'number')
        SMOKING = display_input('Smoking (1 = Yes; 0 = No)', 'Enter 1 if the person smokes, otherwise 0', 'SMOKING', 'number')
        YELLOW_FINGERS = display_input('Yellow Fingers (1 = Yes; 0 = No)', 'Enter 1 if the person has yellow fingers, otherwise 0', 'YELLOW_FINGERS', 'number')
        ANXIETY = display_input('Anxiety (1 = Yes; 0 = No)', 'Enter 1 if the person has anxiety, otherwise 0', 'ANXIETY', 'number')
        PEER_PRESSURE = display_input('Peer Pressure (1 = Yes; 0 = No)', 'Enter 1 if the person is under peer pressure, otherwise 0', 'PEER_PRESSURE', 'number')
        CHRONIC_DISEASE = display_input('Chronic Disease (1 = Yes; 0 = No)', 'Enter 1 if the person has a chronic disease, otherwise 0', 'CHRONIC_DISEASE', 'number')
        FATIGUE = display_input('Fatigue (1 = Yes; 0 = No)', 'Enter 1 if the person experiences fatigue, otherwise 0', 'FATIGUE', 'number')
        ALLERGY = display_input('Allergy (1 = Yes; 0 = No)', 'Enter 1 if the person has allergies, otherwise 0', 'ALLERGY', 'number')
        WHEEZING = display_input('Wheezing (1 = Yes; 0 = No)', 'Enter 1 if the person experiences wheezing, otherwise 0', 'WHEEZING', 'number')
        ALCOHOL_CONSUMING = display_input('Alcohol Consuming (1 = Yes; 0 = No)', 'Enter 1 if the person consumes alcohol, otherwise 0', 'ALCOHOL_CONSUMING', 'number')
        COUGHING = display_input('Coughing (1 = Yes; 0 = No)', 'Enter 1 if the person experiences coughing, otherwise 0', 'COUGHING', 'number')
        SHORTNESS_OF_BREATH = display_input('Shortness Of Breath (1 = Yes; 0 = No)', 'Enter 1 if the person experiences shortness of breath, otherwise 0', 'SHORTNESS_OF_BREATH', 'number')
        SWALLOWING_DIFFICULTY = display_input('Swallowing Difficulty (1 = Yes; 0 = No)', 'Enter 1 if the person has difficulty swallowing, otherwise 0', 'SWALLOWING_DIFFICULTY', 'number')
        CHEST_PAIN = display_input('Chest Pain (1 = Yes; 0 = No)', 'Enter 1 if the person experiences chest pain, otherwise 0', 'CHEST_PAIN', 'number')

        if st.button("Lung Cancer Test Result"):
            try:
                input_values = [
                    float(GENDER), float(AGE), float(SMOKING),
                    float(YELLOW_FINGERS), float(ANXIETY), float(PEER_PRESSURE),
                    float(CHRONIC_DISEASE), float(FATIGUE), float(ALLERGY),
                    float(WHEEZING), float(ALCOHOL_CONSUMING), float(COUGHING),
                    float(SHORTNESS_OF_BREATH), float(SWALLOWING_DIFFICULTY), float(CHEST_PAIN)
                ]
                lungs_prediction = models['lung_cancer'].predict([input_values])
                lungs_diagnosis = "The person has lung cancer disease" if lungs_prediction[0] == 1 else "The person does not have lung cancer disease"
                st.success(lungs_diagnosis)
                text_to_speech(lungs_diagnosis)
            except ValueError:
                st.error("Please enter valid numbers for all fields")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # Hypo-Thyroid Prediction Page
    elif selected == "Hypo-Thyroid Prediction":
        st.title("Hypo-Thyroid Prediction")
        st.write("Enter the following details to predict hypo-thyroid disease:")

        age = display_input('Age', 'Enter age of the person in years', 'age', 'number')
        sex = display_input('Sex (1 = Male; 0 = Female)', 'Enter 1 for male or 0 for female', 'sex', 'number')
        on_thyroxine = display_input('On Thyroxine (1 = Yes; 0 = No)', 'Enter 1 if the person is on thyroxine, otherwise 0', 'on_thyroxine', 'number')
        tsh = display_input('TSH Level', 'Enter thyroid stimulating hormone level in mIU/L', 'tsh', 'number')
        t3_measured = display_input('T3 Measured (1 = Yes; 0 = No)', 'Enter 1 if T3 was measured, otherwise 0', 't3_measured', 'number')
        t3 = display_input('T3 Level', 'Enter triiodothyronine level in ng/dL', 't3', 'number')
        tt4 = display_input('TT4 Level', 'Enter total thyroxine level in μg/dL', 'tt4', 'number')

        if st.button("Thyroid Test Result"):
            try:
                input_values = [
                    float(age), float(sex), float(on_thyroxine),
                    float(tsh), float(t3_measured), float(t3), float(tt4)
                ]
                thyroid_prediction = models['thyroid'].predict([input_values])
                thyroid_diagnosis = "The person has Hypo-Thyroid disease" if thyroid_prediction[0] == 1 else "The person does not have Hypo-Thyroid disease"
                st.success(thyroid_diagnosis)
                text_to_speech(thyroid_diagnosis)
            except ValueError:
                st.error("Please enter valid numbers for all fields")
            except Exception as e:
                st.error(f"Error making prediction: {e}")