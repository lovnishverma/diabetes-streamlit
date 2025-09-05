---
title: Diabetes
emoji: 🏃
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
short_description: This AI-powered Diabetes Prediction App uses a trained Model

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Diabetes Prediction App


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6474405f90330355db146c76/kExsw3b0FM_zqJ5yILQwq.png)


Streamlit Diabetes Prediction App
---------------------------------
- Trains/loads a RandomForest model
- Accepts patient input for diabetes prediction
- Saves each prediction to an audit log (local + Hugging Face dataset repo)
- Displays the 5 most recent predictions


---


# View Logs

https://huggingface.co/datasets/LovnishVerma/diabetes-logs/blob/main/audit_log.csv

https://huggingface.co/datasets/LovnishVerma/diabetes-logs/raw/main/audit_log.csv

# Live View

https://lovnishverma-diabetes.hf.space/