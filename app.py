import streamlit as st
import pandas as pd
import plotly.express as px
import urllib.request
import os
import re
import requests
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Multi-DB GWAS Aggregator", layout="wide", page_icon="🌐")

st.title("🌐 Multi-Database GWAS Federated Aggregator")
st.markdown("A sandbox environment to explore cross-database GWAS catalogs. Use diverse sources for discovery, and unlock direct summary stats comparison via OpenGWAS Core.")

# --- SECRETS CONFIGURATION CHECK ---
try:
    api_token = st.secrets["OPENGWAS_TOKEN"]
except Exception:
    st.error("⚠️ API configuration is missing. Please set 'OPENGWAS_TOKEN' in Streamlit Secrets.")
    st.stop()

headers = {"Authorization": f"Bearer {api_token}"}

# --- DATA LOADING ENGINES ---

# Engine 1: EBI GWAS Catalog (Local TSV Cached)
@st.cache_data
def load_ebi_data():
    url = "https://www.ebi.ac.uk/gwas/api/search/downloads/studies/v1.0.2.1"
    local_file = "gwas_data.tsv"
    if not os.path.exists(local_file):
        with st.spinner('Downloading EBI GWAS Catalog database...'):
            urllib.request.urlretrieve(url, local_file)
    df = pd.read_csv(local_file, sep='\t', low_memory=False)
    df.columns = df.columns.str.strip()
    date_col = [col for col in df.columns if 'date' in col.lower()][0]
    df['Extract_Year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
    def get_n(text):
        nums = re.findall(r'[0-9]+(?:,[0-9]+)*', str(text))
        return int(nums[0].replace(',', '')) if nums else 0
    df['N_Size'] = df['INITIAL SAMPLE SIZE'].apply(get_n)
    return df

# Engine 2: OpenGWAS Active Core Inventory API
@st.cache_data
def load_opengwas_inventory():
    res = requests.get("https://api.opengwas.io/api/gwasinfo", headers=headers)
    if res.status_code == 200:
        data = res.json()
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient='index')
        else:
            df = pd.DataFrame(data)
            
        if 'id' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'id'})
            
        if 'year' in df.columns:
            df['Extract_Year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2020).astype(int)
        else:
            df['Extract_Year'] = 2020
            
        if 'sample_size' in df.columns:
            df['N_Size'] = pd.to_numeric(df['sample_size'], errors='coerce').fillna(0).astype(int)
        else:
            df['N_Size'] = 0
            
        for col in ['author', 'trait', 'population']:
            if col not in df.columns:
                df[col] = "Unknown"
                
        return df
    return pd.DataFrame()

# --- SIDEBAR: DATABASE CHANGER ---
st.sidebar.header("🌐 Data Source Gateway")
db_source = st.sidebar.selectbox(
    "Select Active Data Provider:",
    ["EBI GWAS Catalog (Comprehensive)", "OpenGWAS Core Inventory (Live API)"]
)

# Load data dynamically based on selection
if db_source == "EBI GWAS Catalog (Comprehensive)":
    df = load_ebi_data()
    trait_col = [col for col in df.columns if 'trait' in col.lower() or 'disease' in col.lower()][0]
    sample_col = 'INITIAL SAMPLE SIZE'
    author_col = 'FIRST AUTHOR'
    id_col = 'STUDY ACCESSION'
else:
    with st.spinner("Fetching active OpenGWAS core inventory..."):
        df = load_opengwas_inventory()
    trait_col = 'trait'
    sample_col = 'population' 
    author_col = 'author'
    id_col = 'id'

# --- FILTER FORM PANEL ---
if not df.empty:
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.header("🛠️ Specialized Filters")
        st.info(f"Active Provider: **{db_source.split()[0]}**")
        
        with st.form("filter_form"):
            st.markdown("### Disease / Trait")
            all_traits = sorted(df[trait_col].dropna().unique().tolist())
            selected_trait = st.selectbox("Select Trait", ["All"] + all_traits)
            
            st.markdown("### Ancestry / Population")
            major_ancestries = ["European", "African", "East Asian", "South Asian", "Admixed American", "Hispanic", "Latino"]
            selected_ancestries = st.multiselect("Quick-Select Cohort", major_ancestries)
            searched_ancestry = st.text_input("Manual Search", placeholder="e.g., Finnish, Japanese")
            
            if db_source.startswith("EBI"):
                strict_ancestry = st.checkbox("Strict Ancestry (Exclude mixed cohorts)", value=False)
            else:
                strict_ancestry = False
                
            st.markdown("### Metrics & Timeline")
            min_year = int(df['Extract_Year'].min()) if pd.notnull(df['Extract_Year'].min()) else 2000
            max_year = int(df['Extract_Year'].max()) if pd.notnull(df['Extract_Year'].max()) else 2026
            selected_year = st.slider("Minimum Publication Year", min_year, max_year, min_year)
            
            st.markdown("### Sorting")
            sort_by = st.selectbox("Order Results By:", ["Sample Size (High to Low)", "Publication Year (Newest)", "First Author (A-Z)"])
            
            submitted = st.form_submit_button("🚀 Query Database")
            
        if submitted:
            st.session_state["sandbox_submitted"] = True
            
    # --- RESULTS PANEL ---
    with col2:
        st.subheader("📊 Consolidated View")
        
        if st.session_state.get("sandbox_submitted", False):
            res = df.copy()
            
            if selected_trait != "All":
                res = res[res[trait_col].astype(str) == selected_trait]
                
            if selected_ancestries:
                pattern = '|'.join([anc.lower() for anc in selected_ancestries])
                res = res[res[sample_col].astype(str).str.contains(pattern, case=False, na=False)]
                
            if searched_ancestry:
                res = res[res[sample_col].astype(str).str.contains(searched_ancestry, case=False, na=False)]
                
            if strict_ancestry and db_source.startswith("EBI"):
                res = res[~res[sample_col].astype(str).str.contains('admixed|mixed|cross-population', case=False, na=False)]
                if selected_ancestries:
                    exclude_list = [a for a in major_ancestries if a not in selected_ancestries]
                    if exclude_list:
                        pattern_exclude = '|'.join([anc.lower() for anc in exclude_list])
                        res = res[~res[sample_col].astype(str).str.contains(pattern_exclude, case=False, na=False)]
                        
            res = res[res['Extract_Year'] >= selected_year]
            
            if sort_by == "Sample Size (High to Low)":
                res = res.sort_values(by='N_Size', ascending=False)
            elif sort_by == "Publication Year (Newest)":
                res = res.sort_values(by='Extract_Year', ascending=False)
            elif sort_by == "First Author (A-Z)":
                res = res.sort_values(by=author_col, ascending=True)
                
            if not res.empty:
                tab_charts, tab_map = st.tabs(["📈 Statistical Breakdown", "🌍 Federated Cohort Map"])
                
                with tab_charts:
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        fig_pie = px.pie(res.head(20), names=author_col, values='N_Size', 
                                         title="Top Cohorts Distribution by Sample Size (N)",
                                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with v_col2:
                        year_counts = res['Extract_Year'].value_counts().sort_index().reset_index()
                        year_counts.columns = ['Year', 'Count']
                        fig_line = px.line(year_counts, x='Year', y='Count', title="Studies Cataloged Over Time", markers=True)
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                with tab_map:
                    st.markdown("##### 📍 Harmonized Spatial Variant Engine")
                    global_geo_db = {
                        'UK': {'lat': 55.3781, 'lon': -3.4360, 'lbl': 'United Kingdom'},
                        'United Kingdom': {'lat': 55.3781, 'lon': -3.4360, 'lbl': 'United Kingdom'},
                        'British': {'lat': 55.3781, 'lon': -3.4360, 'lbl': 'United Kingdom'},
                        'European': {'lat': 50.1109, 'lon': 8.6821, 'lbl': 'European Core'},
                        'Eur': {'lat': 50.1109, 'lon': 8.6821, 'lbl': 'European Core'},
                        'Finland': {'lat': 61.9241, 'lon': 25.7482, 'lbl': 'Finland'},
                        'Finnish': {'lat': 61.9241, 'lon': 25.7482, 'lbl': 'Finland'},
                        'Japan': {'lat': 36.2048, 'lon': 138.2529, 'lbl': 'Japan'},
                        'Japanese': {'lat': 36.2048, 'lon': 138.2529, 'lbl': 'Japan'},
                        'East Asian': {'lat': 34.0479, 'lon': 100.6197, 'lbl': 'East Asian Core'},
                        'Eas': {'lat': 34.0479, 'lon': 100.6197, 'lbl': 'East Asian Core'},
                        'Korea': {'lat': 35.9078, 'lon': 127.7669, 'lbl': 'Korea'},
                        'Korean': {'lat': 35.9078, 'lon': 127.7669, 'lbl': 'Korea'},
                        'United States': {'lat': 37.0902, 'lon': -95.7129, 'lbl': 'United States'},
                        'US': {'lat': 37.0902, 'lon': -95.7129, 'lbl': 'United States'},
                        'USA': {'lat': 37.0902, 'lon': -95.7129, 'lbl': 'United States'},
                        'African': {'lat': 1.6508, 'lon': 22.5644, 'lbl': 'African Core'},
                        'Afr': {'lat': 1.6508, 'lon': 22.5644, 'lbl': 'African Core'}
                    }
                    
                    map_rows = []
                    for idx, row in res.iterrows():
                        sample_text = str(row[sample_col])
                        n_size = row['N_Size']
                        for key, coord in global_geo_db.items():
                            if re.search(r'\b' + re.escape(key) + r'\b', sample_text, re.IGNORECASE):
                                jitter = 1.8
                                map_rows.append({
                                    'Region': coord['lbl'],
                                    'Latitude': coord['lat'] + np.random.uniform(-jitter, jitter),
                                    'Longitude': coord['lon'] + np.random.uniform(-jitter, jitter),
                                    'N_Size': n_size,
                                    'Trait': str(row[trait_col]),
                                    'Author': str(row[author_col])
                                })
                                break
                                
                    map_data = pd.DataFrame(map_rows)
                    if not map_data.empty:
                        map_data['size_normalized'] = np.sqrt(map_data['N_Size'].clip(lower=1))
                        fig_map = px.scatter_geo(
                            map_data, lat="Latitude", lon="Longitude", size="size_normalized",
                            hover_name="Region", hover_data={'N_Size': ':,', 'Trait': True, 'Author': True, 'size_normalized': False},
                            size_max=15, projection="natural earth", color="N_Size",
                            color_continuous_scale=px.colors.sequential.Plasma, template="plotly_white", opacity=0.75
                        )
                        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, geo=dict(showland=True, landcolor="#F4F4F4"))
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.info("🗺️ No standardized geographic mapping could be inferred for this subset.")
                        
                st.markdown("---")
                st.write(f"**Total Consolidated Rows Found:** {len(res)}")
                
                standardized_df = pd.DataFrame({
                    'Database Source': db_source.split()[0],
                    'Target ID': res[id_col].astype(str),
                    'Trait / Phenotype': res[trait_col].astype(str),
                    'Cohort Description': res[sample_col].astype(str),
                    'Sample Size (N)': res['N_Size'],
                    'First Author': res[author_col].astype(str),
                    'Year': res['Extract_Year']
                }).head(100)
                
                # Akıllı yönlendirme için çoklu seçim modu aktif
                selection = st.dataframe(
                    standardized_df, use_container_width=True, on_select="rerun", selection_mode="multi-row"
                )
                
                selected_rows = selection.get("selection", {}).get("rows", [])
                
                # 🛠️ STRATEJİK DEĞİŞİKLİK: Seçilen Veritabanına Göre Buton Davranışı
                if db_source.startswith("OpenGWAS"):
                    if len(selected_rows) == 2:
                        id1 = standardized_df.iloc[selected_rows[0]]['Target ID']
                        id2 = standardized_df.iloc[selected_rows[1]]['Target ID']
                        st.success(f"Locked OpenGWAS IDs: **{id1}** and **{id2}**")
                        if st.button("🚀 Run Cross-Population Portability Analysis", type="primary"):
                            st.session_state["auto_study_1"] = id1
                            st.session_state["auto_study_2"] = id2
                            st.success("Redirecting to comparison portal...")
                            # st.switch_page("pages/2_⚖️_Compare_GWAS.py")
                    elif len(selected_rows) > 2:
                        st.warning("⚠️ Please select exactly 2 studies for comparison.")
                    else:
                        st.info("💡 Select exactly TWO rows from the table above to unlock the 'Compare' dashboard bridge.")
                
                else:
                    # EBI Seçildiyse sadece veri izleme modudur, buton çıkmaz yönlendirme yapılmaz
                    st.info("ℹ️ **Discovery Mode Active:** EBI GWAS Catalog is currently optimized for cohort distribution discovery and metadata visualization. To execute down-stream variant alignment (Beta/EAF Correlation), please switch the active data provider to **OpenGWAS Core Inventory** in the sidebar.")
            else:
                st.warning("⚠️ No records matched the constraints inside the selected database engine.")
        else:
            st.info("👈 Choose your master Data Provider on the left and hit 'Query' to initiate unified view.")
