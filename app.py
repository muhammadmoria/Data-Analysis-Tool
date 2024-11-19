import pandas as pd
import plotly.express as px
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Data Analysis Portal",
    page_icon="📈",
    layout="wide"
)

st.title(":rainbow[Data Analytics Portal]")
st.subheader(":gray[Explore Data With Ease]", divider="rainbow")

# File Uploader
file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if file is not None:
    try:
        # Read file
        if file.name.endswith("csv"):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)

        st.dataframe(data)
        st.info("Data uploaded successfully!", icon="✅")
        
        # Basic Dataset Information
        st.subheader(":rainbow[Basic Information About the Dataset]", divider="rainbow")
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Top & Bottom Rows", "Data Types", "Columns"])
        
        with tab1:
            st.write(f"**Rows:** {data.shape[0]} | **Columns:** {data.shape[1]}")
            st.subheader(":gray[Statistical Summary]")
            st.dataframe(data.describe())
        
        with tab2:
            st.subheader(":gray[Top Rows]")
            toprows = st.slider("Select number of rows to display (Top)", 1, min(10, data.shape[0]), key="toprows")
            st.dataframe(data.head(toprows))
            st.subheader(":gray[Bottom Rows]")
            bottomrows = st.slider("Select number of rows to display (Bottom)", 1, min(10, data.shape[0]), key="bottomrows")
            st.dataframe(data.tail(bottomrows))
        
        with tab3:
            st.subheader(":gray[Data Types]")
            st.dataframe(data.dtypes)
        
        with tab4:
            st.subheader(":gray[Columns]")
            st.dataframe(pd.DataFrame({"Columns": data.columns}))

        # Value Counts Section
        st.subheader(":rainbow[Column Values Count]", divider="rainbow")
        with st.expander("Value Counts"):
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Choose Column for Value Counts", options=data.columns)
            with col2:
                topr = st.number_input("Top Rows to Display", min_value=1, value=5, step=1)

            if st.button("Generate Value Counts"):
                try:
                    result = data[column].value_counts().reset_index().head(topr)
                    result.columns = [column, "Count"]
                    st.dataframe(result)
                    
                    # Bar Chart
                    st.subheader(":gray[Bar Chart]")
                    fig_bar = px.bar(result, x=column, y="Count", text="Count", template="plotly_white")
                    st.plotly_chart(fig_bar)
                    
                    # Line Chart
                    st.subheader(":gray[Line Chart]")
                    fig_line = px.line(result, x=column, y="Count", template="plotly_white", markers=True)
                    st.plotly_chart(fig_line)
                    
                    # Pie Chart
                    st.subheader(":gray[Pie Chart]")
                    fig_pie = px.pie(result, values="Count", names=column, template="plotly_white")
                    st.plotly_chart(fig_pie)
                except Exception as e:
                    st.error(f"An error occurred while generating value counts: {e}")

        # Group By Section
        st.subheader(":rainbow[Group By: Simplify Your Analysis]", divider="rainbow")
        with st.expander("Group By Your Columns"):
            col1, col2, col3 = st.columns(3)

            with col1:
                groupby_cols = st.multiselect("Choose Columns to Group By", options=data.columns)
            with col2:
                operation_col = st.selectbox("Choose Column for Aggregation", options=data.columns)
            with col3:
                agg_func = st.selectbox("Choose Aggregation Function", options=["sum", "mean", "median", "max", "min"])

            if groupby_cols:
                try:
                    result = data.groupby(groupby_cols).agg(Grouped=(operation_col, agg_func)).reset_index()
                    st.dataframe(result)
                    
                    # Visualization
                    st.subheader(":gray[Visualize Grouped Data]")
                    graph_type = st.selectbox("Choose Graph Type", options=["Line Chart", "Bar Chart", "Scatter Chart", "Pie Chart", "Sunburst Chart"])
                    
                    if graph_type == "Line Chart":
                        x_axis = st.selectbox("X-axis", options=result.columns)
                        y_axis = st.selectbox("Y-axis", options=result.columns)
                        color = st.selectbox("Color (Optional)", options=[None] + list(result.columns))
                        fig = px.line(result, x=x_axis, y=y_axis, color=color, markers=True, template="plotly_white")
                        st.plotly_chart(fig)
                    
                    elif graph_type == "Bar Chart":
                        x_axis = st.selectbox("X-axis", options=result.columns)
                        y_axis = st.selectbox("Y-axis", options=result.columns)
                        color = st.selectbox("Color (Optional)", options=[None] + list(result.columns))
                        fig = px.bar(result, x=x_axis, y=y_axis, color=color, template="plotly_white")
                        st.plotly_chart(fig)

                    elif graph_type == "Scatter Chart":
                        x_axis = st.selectbox("X-axis", options=result.columns)
                        y_axis = st.selectbox("Y-axis", options=result.columns)
                        color = st.selectbox("Color (Optional)", options=[None] + list(result.columns))
                        fig = px.scatter(result, x=x_axis, y=y_axis, color=color, template="plotly_white")
                        st.plotly_chart(fig)
                    
                    elif graph_type == "Pie Chart":
                        values = st.selectbox("Values", options=result.columns)
                        names = st.selectbox("Names", options=result.columns)
                        fig = px.pie(result, values=values, names=names, template="plotly_white")
                        st.plotly_chart(fig)
                    
                    elif graph_type == "Sunburst Chart":
                        path = st.multiselect("Path", options=result.columns)
                        values = st.selectbox("Values", options=result.columns)
                        fig = px.sunburst(result, path=path, values=values, template="plotly_white")
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred during grouping or visualization: {e}")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.warning("Please upload a file to proceed.")


# Footer Section
st.markdown("""
---
## 📄 About Me  
Hi! I'm **Muhammad Dawood**, a data scientist passionate about Machine Learning, NLP, and building intelligent solutions.  

### 🌐 Connect with Me:  
- [GitHub](https://github.com/muhammadmoria)  
- [LinkedIn](https://www.linkedin.com/in/muhammaddawood361510306/)  
- [Portfolio](https://muhammadmoria.github.io/portfolio-new/)  
""")
