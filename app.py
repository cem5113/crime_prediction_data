import streamlit as st
import pandas as pd

st.title("🧪 Streamlit Testi")

df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": ["x", "y", "z"]
})

st.write("📋 Örnek Veri:")
st.dataframe(df)

if st.button("Test Butonu"):
    st.success("✅ Butona tıklandı!")
