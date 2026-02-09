import streamlit as st
import plotly.io as pio
import gdown
import os
from PIL import Image

gdrive_dict = {'area_crimetype_heatmap.json':'1TJiv9xgoa6Kaut-Oi8vL8-T2lngMB6zi',
               'diurnal_heatmap.json':'1RsLPtfXTXiMNHRWcYpHqN45MpPCfXPpD',
               'crime_choropleth_map.json':'10zDHrCXcWuwe8MtW1ctKf5FPtNS1hLTp',
               'time_series_seasonality.json':'1l5-chpbi_n3J8yAUytzF8mJD5jqshURA',
               'top_crime_annual.json':'1nV7WUgQHpmK-DGagmm5sGc5bOnFFeyco',
               'arrest_rate.png':'1U6JqhoYsaPMThrGGLOm3zpOH2swAk4oI'}


file_name = list(gdrive_dict.keys())

if st.button("Reload charts"):
    st.cache_data.clear()

@st.cache_data
def check_file(file_id,file_name):
    file_type = file_name.split('.')[1]

    if os.path.exists(file_name):
        pass
    else:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, file_name, quiet=False)
        
    if file_type == 'json':
        file_read = pio.read_json(file_name)
    else:
        file_read = Image.open(file_name)

    return file_read



fig_heatmap_area_crime = check_file(gdrive_dict[file_name[0]],file_name[0])
fig_heatmap_diurnal = check_file(gdrive_dict[file_name[1]],file_name[1])
fig_choropleth = check_file(gdrive_dict[file_name[2]],file_name[2])
fig_time_series = check_file(gdrive_dict[file_name[3]],file_name[3])
fig_top_crime = check_file(gdrive_dict[file_name[4]],file_name[4])
fig_arrest_rate = check_file(gdrive_dict[file_name[5]],file_name[5])

# ========== MAIN PAGE ==========
st.title("📊 IT5006 Group 23 - Chicago Crime")
st.write("An exploratory data analysis was conducted towards Chicago's Crime dataset (2015-2025) provided by the open-source Chicago Data Portal. " \
"In this analysis, the team's main focus is to gain insights of how crimes behave in Chicago. The insights that the team would try to uncover are where crimes occur, " \
"what types of crime occur, and when do they happen.")

st.header("Crime Density Choropleth Map of Chicago")
st.write("A choropleth map of Chicago's crime density (crime/km²) is plotted to visualize the spatial distribution." \
"By adjusting the year filter, it is apparent that areas with initially high crime density continue to experience more crime than lower-density areas in the following years. The **central and near-shore areas of Chicago have consistent high crime density**.")
st.plotly_chart(fig_choropleth, width='stretch')

st.header("Crime Occurence Time Series Seasonality")
st.write("When plotted into a time series, it can be seen that crime occurence have a seasonality pattern. The most notable seasonality pattern is when the crime " \
"occurence is plotted by Months. Crimes are at their lowest during the **first few months of the year**  and it gradually increases toward the middle of the year, **peaking in July and August most of the time**. Finally, it continues to decrease by the end of" \
" the year and the pattern continues for the following years.")
st.plotly_chart(fig_time_series, width='stretch')

st.header("Highest Crime in Chicago Annually")
st.write("There are many crime classifications from the dataset and to identify each crime type will become troublesome since some can be classified as noise if it does not bring any value into the EDA. To ensure that " \
"the crime types are consistent, the crime types are ranked and the top 10 is the main focus of the EDA. It appears that the crime types are consistent throughout the years " \
"with **Theft, Battery, and Criminal Damage** ranking the highest while **others remain in the top 10 but interchange in ranking**.")
st.plotly_chart(fig_top_crime, width='stretch')

st.header("Crime Heatmap of Chicago Community Area")
st.write("To understand the amount of crimes that happened in each Chicago Community area, a heatmap was made. Although it only covers the top 10 community area with the highest crime occurence, "
"it still provides a guidance for the EDA. For example:")
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;• Theft frequently occurs in Austin, Near North Side, Near West Side, Loop, and West Town")
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;• Battery usually happens in Austin, South Shore North Lawndale, and Humboldt Park") 
st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;• Narcotic crimes are highest in North Lawndale and Humboldt Park.")
st.write("This suggests that **certain crime types are more prevalent in some areas more than others**.")
st.plotly_chart(fig_heatmap_area_crime, width='stretch')

st.header("Heatmap of Diurnal Crime Occurence")
st.write("The following heatmap provides information how crime types are distributed throughout the times of day and the days of week. From the heatmap, " \
"it is understood that crime rates vary by type depending on the time of day. For example, deceptive crimes usually occur in the middle of the afternoon while criminal " \
"damage usually happens from the evening until midnight.")
st.plotly_chart(fig_heatmap_diurnal, width='stretch')

st.header("Crime Arrest Rate")
st.write("With the amount of crimes that are happening in Chicago, it is important to understand if the crimes are handled properly. Unfortunately, it was " \
"discovered that the arrest rate for the top 10 most occuring crime are in the lower ranks based on the following bar chart. This signifies the importance " \
"of estimating and predicting where and when crimes can happen. This will change how crime policing can transform from a reactive approach to a preventive approach.")
st.image(fig_arrest_rate)
# ========== SUMMARY PAGE ==========

st.title(f"📄Summary")
st.write("The exploratory data analysis conducted on Chicago's data crime from 2015 to 2025 gained meaningful insights. It was discovered that not only do crimes have " \
"a seasonality pattern based on temporal trends but certain crime types are more frequent in certain parts of Chicago compared to other neighboring areas. It was later revealed in the end that " \
"law enforcement authorities still have difficulties in making arrests especially for the most occuring crime types in Chicago namely Theft, Battery, and Criminal Damage. From these discoveries, the team will move forward " \
"in possibly developing an ML model that can predict where and when potential crimes may happen, giving law enforcements a shift in strategy from reactive to preventive action.")

