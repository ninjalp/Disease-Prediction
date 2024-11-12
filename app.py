import streamlit as st
import torch
import torch.nn as nn
import pickle


class PrognosisModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PrognosisModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        return x

with open('prognosis_model_with_class_names.pkl', 'rb') as f:
    model_data = pickle.load(f)

with open("symptoms_list_turkish.pkl", "rb") as f:
    symptoms = pickle.load(f)


model = PrognosisModel(input_size=model_data['input_size'], num_classes=model_data['num_classes'])
model.load_state_dict(model_data['model_state_dict'])
model.eval()  


class_names = model_data['class_names']
st.title("🩺 Hastalık Teşhis Uygulaması")
st.markdown("Bu uygulama belirtilerinize göre olası hastalıkları tahmin eder. Lütfen mevcut belirtilerinizi seçin ve **'Teşhis Et'** butonuna basın.")
st.markdown("---")
st.subheader("Belirtilerinizi Girin")
col1, col2 = st.columns(2)  
# Türkçe belirtiler listesini yükleme
with open("symptoms_list_turkish.pkl", "rb") as f:
    symptoms_turkish = pickle.load(f)
symptom_inputs = []
columns = st.columns(3)  # 3 sütun oluştur

for idx, symptom in enumerate(symptoms_turkish):
    col = columns[idx % 3]  
    with col:
        value = st.selectbox(f"{symptom}", [0, 1], index=0, format_func=lambda x: "Yok" if x == 0 else "Var", key=f"selectbox_{idx}")
        symptom_inputs.append(value)

st.markdown("---")


if st.button("Teşhis Et", key="diagnosis_button"):
    
    input_tensor = torch.tensor([symptom_inputs], dtype=torch.float32)
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    
 
    st.subheader("Teşhis Sonucu:")
    st.markdown(f"**Tahmin Edilen Hastalık:** {class_names[predicted_class]}")
    st.success(f"Modelin tahmin ettiği teşhis: **{class_names[predicted_class]}**")