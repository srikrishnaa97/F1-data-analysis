import streamlit as st
import plotly.express as px
import fastf1
import fastf1.plotting
import streamlit.components.v1 as components
import datetime as dt
import pandas as pd
import numpy as np
from plot_functions import plot_speed_segments, telemetry, get_driver_laps, lap_times_plot
from basic_functions import get_results_df_as_html

st.set_page_config(
    page_title="F1 Data Analysis",
    page_icon="./images/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': '',
        # 'Report a bug': "",
        'About': "### Thanks for checking out the F1 Data Analysis App!"
    }
)

col1, mid, col2 = st.columns([1, 1, 20])
with col1:
    st.image('./images/favicon.png', width=100)
with col2:
    st.title('Formula 1 Data Analysis')


@st.cache_resource(max_entries=1,show_spinner='Fetching session...')
def get_session_data(year, gp, session):
    # _, cache_size = fastf1.Cache.get_cache_info()
    # if cache_size >= 3e8:
    #     fastf1.Cache.clear_cache()
    data = fastf1.get_session(year, gp, session)
    data.load()
    return data


@st.cache_resource(max_entries=1,show_spinner='Fetching schedule...')
def get_event_schedule_data(year):
    schedule = fastf1.get_event_schedule(year)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    today = dt.datetime.now()
    schedule['Days from Today'] = (schedule['EventDate'] - today).dt.days
    return schedule


@st.cache_resource(max_entries=1,show_spinner='Fetching event...')
def get_event_data(year, gp):
    event = fastf1.get_event(year, gp)
    return event


# Sidebar

current_year = dt.datetime.now().year

events = get_event_schedule_data(current_year)
past_events = events[events['Days from Today'] < 0]

if len(past_events) == 0:
    past_events = get_event_schedule_data(current_year - 1)
    current_year -= 1

year = st.sidebar.selectbox(
    "Year",
    np.arange(current_year, 2020, -1)
)
events = get_event_schedule_data(year)
past_events = events[events['Days from Today'] < 0]
latest_event = len(past_events) - 1

gp = st.sidebar.selectbox(
    "Grand Prix",
    events.sort_values('EventDate').EventName.to_list(),
    index=latest_event
)
event = get_event_data(year, gp)
session = st.sidebar.selectbox(
    "Session",
    [event.get_session_name(n) for n in range(5, 0, -1)]
)
try:
    data = get_session_data(year, gp, session)
    all_drivers = list(data.results.Abbreviation.unique())
    drivers = st.sidebar.multiselect(
        "Driver(s)",
        all_drivers, 
        default=data.laps.groupby('Driver').LapTime.min().sort_values().reset_index()['Driver'].iloc[:2].to_list(),
        max_selections=3
    )
    if len(drivers) == 0:
        drivers = all_drivers[:3]

    display_data_flag = 1

except:
    display_data_flag = 0
    st.warning("That event hasn't happened yet or doesn't exist! Please try again!", icon="⚠️")

footer = """<style>

.footer {
position: fixed;
bottom: 0;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Srikrishnaa J</p>
</div>
"""
st.sidebar.markdown(footer, unsafe_allow_html=True)

event_data = {'year':year,'gp':gp,'session':session}


# Tabs
if display_data_flag:
    tab1, tab3, tab6 = st.tabs(["Results", "Track Dominance", "Telemetry"])

    #       Tab 1
    with tab1:
        st.header(f'{year} {gp} {session} Results')
        html = get_results_df_as_html(data)
        st.markdown(html, unsafe_allow_html=True)

    with tab3:
        subtab1, subtab2 = st.tabs(["Fastest Lap", "Full session"])
        with subtab1:
            st.header(f'{year} {gp} {session} Track Dominance Fastest Lap')
            fig1, kpi_dict = plot_speed_segments(data, drivers, fastest_lap=True)
            if isinstance(fig1,int):
                st.warning(f"Error loading data for {kpi_dict}! Please remove {kpi_dict} from driver selection and try again!", icon="⚠️")
            else:
                for i, col in enumerate(st.columns(len(kpi_dict))):
                    with col:
                        driver = list(kpi_dict.keys())[i]
                        value = kpi_dict[driver]
                        st.metric(label=driver, value=value)
                        st.markdown(f'<h4 style="color:{fastf1.plotting.driver_color(driver)}">{driver}</h4>',
                                unsafe_allow_html=True)
                
                for i, col in enumerate(st.columns(2)):
                    with col:
                        if i == 0:
                            st.plotly_chart(fig1, theme="streamlit", use_container_width=False)
                        else:
                            st.empty()
        with subtab2:
            st.header(f'{year} {gp} {session} Track Dominance Full Session')
            fig2, kpi_dict = plot_speed_segments(data, drivers, fastest_lap=False)
            if isinstance(fig2,int):
                st.warning(f"Error loading data for {kpi_dict}! Please remove {kpi_dict} from driver selection and try again!", icon="⚠️")
            else:
                st.plotly_chart(fig2,theme="streamlit",use_container_width=False)
    
    #       Tab 6
    with tab6:
        st.header(f'{year} {gp} {session}')
        fig1 = lap_times_plot(data,drivers) 
        if isinstance(fig1,str):
            st.warning(f"Error loading data for {fig1}! Please remove {fig1} from driver selection and try again!", icon="⚠️")
        else:
            st.header(f'Lap Times')
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

            # Get lap input
        
            def reset_lap_numbers(driver):
                return None
            
            lap_numbers = [None]*len(drivers)

            vals = {}
            st.header("Lap Select")
            for i, col in enumerate(st.columns(len(drivers))):
                with col: 
                    st.markdown(f'<h4 style="color:{fastf1.plotting.driver_color(drivers[i])}">{drivers[i]}</h4>',
                            unsafe_allow_html=True)
                    laps = get_driver_laps(data,drivers[i])
                    fastest_time = laps.LapTime.min()
                    fastest_lap = laps[laps.LapTime == fastest_time].LapNumber.iloc[0]
                    options = [f"Lap {int(lap['LapNumber'])} | {lap['Compound']}" if lap['LapNumber'] != fastest_lap else f"Lap {int(lap['LapNumber'])} | {lap['Compound']} | Fastest" for _,lap in laps.iterrows()]
                    default = [i for i, value in enumerate(options) if 'Fastest' in value]
                    vals[drivers[i]] = st.selectbox(label=f"Lap Select", options=options, index=int(default[0]), key=f"box_{i}")
                    vals[drivers[i]] = int(vals[drivers[i]].split('|')[0][3:-1])
                            
            lap_nums = vals if len(vals) > 0 else {d: None for d in drivers}
            fig2, fig3, kpi_dict = telemetry(lap_nums,data,drivers)
            if isinstance(fig2,str):
                st.warning(f"Error loading data for {fig2}! Please remove {fig2} from driver selection and try again!", icon="⚠️")
            elif isinstance(fig3,str):
                st.warning(f"Error loading data for {fig3}! Please remove {fig3} from driver selection and try again!", icon="⚠️")
            else:
                for i, col in enumerate(st.columns(len(drivers))):
                    with col:
                        driver = drivers[i]
                        value = kpi_dict[driver]
                        st.metric(label="Sample", value=value, label_visibility='hidden')

                st.header(f'Track Animation')
                _, mid, __ = st.columns([1, 20, 1])
                with mid:
                    st.plotly_chart(fig2, theme="streamlit", use_container_width=False)
                st.header(f'Telemetry')
                st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
