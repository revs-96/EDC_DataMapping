import streamlit as st
import pandas as pd
from xml_loader import parse_source_xml
from trainer import train_pipeline
from predictor import predict_mappings
from persist import save_new_mapping

st.set_page_config(page_title='EDC Mapping', layout='wide')
st.title('EDC Mapping — Exact Mapping & Human-in-the-loop')

tab1, tab2 = st.tabs(['Train Model', 'Predict Mappings'])

# ---------------------- TRAINING ----------------------
with tab1:
    st.header('Train model from Source XML + ViewMapping')
    src_file = st.file_uploader('Upload Source XML (StudyData)', type=['xml'])
    mapping_file = st.file_uploader('Upload ViewMapping XML', type=['xml'])
    
    if st.button('Train'):
        if not src_file or not mapping_file:
            st.error('Upload both Source and ViewMapping XML files to train.')
        else:
            src_bytes = src_file.read()
            map_bytes = mapping_file.read()
            st.info('Training — this may take a while...')
            stats = train_pipeline(src_bytes, map_bytes)
            st.success(f"Training completed. Total samples: {stats['n_samples']}, Positives: {stats['n_pos']}")

# ---------------------- PREDICTION ----------------------
with tab2:
    st.header('Predict mappings for a Source XML')
    pred_file = st.file_uploader('Upload Source XML to predict mappings', type=['xml'])

    if st.button('Predict'):
        if not pred_file:
            st.error('Upload Source XML to predict.')
        else:
            src_bytes = pred_file.read()
            results = predict_mappings(src_bytes)

            confident_rows = []
            unmapped_studies = []

            # Collect exact mappings & unmapped StudyEventOIDs
            for res in results:
                se_oid = res['StudyEventOID']
                confident_rows.extend([
                    {
                        'StudyEventOID': se_oid,
                        'ItemOID': c['ItemOID'],
                        'Target': c['Target'],
                        'Score': c['Score'],
                        'Cosine': c['Cosine']
                    } for c in res['confident']
                ])
                if not res['confident']:  # No exact mapping found
                    unmapped_studies.append(se_oid)

            # Display confident mappings
            if confident_rows:
                st.subheader("Confident Exact Mappings")
                df_confident = pd.DataFrame(confident_rows)
                st.dataframe(df_confident)

            # HITL for unmapped StudyEventOID
            if unmapped_studies:
                st.subheader("Unmapped StudyEvents — Human validation required")
                for se_oid in unmapped_studies:
                    mapping_text = st.text_input(
                        f"Provide mapping text for StudyEventOID: {se_oid}",
                        key=f"mapping_{se_oid}"
                    )
                    if mapping_text:
                        save_new_mapping(se_oid, mapping_text)
                        st.success(f"Mapping added for {se_oid}: {mapping_text}")