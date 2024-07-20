from basic_functions import convert_timedelta_to_time, rotate
import datetime as dt
import pandas as pd
import plotly.express as px
import fastf1.plotting
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_basic_plot_data(data,driver,lap_number):
    if lap_number == None:
        lap_time = data.laps.pick_driver(driver).pick_fastest()['LapTime'].iloc[0]
        return lap_time, data.laps.pick_driver(driver).pick_fastest().get_car_data().add_distance()
    else:
        lap_time = data.laps.pick_driver(driver).pick_lap(lap_number)['LapTime'].iloc[0]
        return lap_time, data.laps.pick_driver(driver).pick_lap(lap_number).get_car_data().add_distance()

def basic_plots(data,drivers,lap_numbers):
    df = pd.DataFrame()
    plots = [
        'Speed',
        'Throttle',
        'DRS',
        'nGear'
    ]

    lap_time = {}
    for i,d in enumerate(drivers):
        lap, driver_df = get_basic_plot_data(data,d,lap_numbers[d])
        driver_df['Time'] = driver_df['Time'] + pd.to_datetime("1970-01-01 00:00:00.000",format="%Y-%m-%d %H:%M:%S.%f") #dt.datetime(1970,1,1,0,0,0,0)
        driver_df['Time'] = driver_df['Time'].apply(lambda x: x.to_pydatetime())
        driver_df['Driver'] = d
        df = pd.concat([df, driver_df], axis=0)

        lap_time[d] = convert_timedelta_to_time(lap)

    # Lap Time
    lap_time = pd.DataFrame(lap_time.values(), columns=['LapTime'], index=lap_time.keys()).sort_values(
        'LapTime').reset_index()
    lap_time.rename({'index': 'Driver'}, axis=1, inplace=True)
    kpi_dict = {}
    for i in range(min(len(lap_time), 5)):
        minutes = int(lap_time['LapTime'].iloc[i] // 60)
        seconds = int(lap_time['LapTime'].iloc[i] % 60)
        milli = int(str(lap_time['LapTime'].iloc[i]).split('.')[-1].strip())
        kpi_dict[lap_time['Driver'].iloc[i]] = f"{minutes:>02d}:{seconds:>02d}.{milli:<03d}"

    # Plots
    fig = make_subplots(rows=len(plots),cols=1,vertical_spacing=0.1,row_heights=[500]*len(plots),subplot_titles=plots)
    for i, p in enumerate(plots):
        for d in drivers:
            subdf = df[df['Driver']==d]#.sort_values('Time')
            fig.add_trace(
                go.Scatter(
                    x=subdf['Time'].to_numpy(),
                    y=subdf[p].to_numpy(),
                    mode="lines",
                    line=dict(
                        color=fastf1.plotting.driver_color(d)
                    ),
                    showlegend=False,
                    name=f"{d},{p}"
                ),
                row=i+1,
                col=1
            )
        fig.update_xaxes(
            tickformat="%M:%S.%f",
            row=i+1,
            col=1
        )
        fig.update_yaxes(
            title=p,
            row=i+1,
            col=1
        )
        
    fig.update_layout(
        height=300*len(plots),
    )
    fig.update_annotations(
        font_size=25
    )
    return fig, kpi_dict

def get_driver_laps(data,driver):
    return data.laps.pick_driver(driver)[['LapNumber','Compound','LapTime']]#.sort_values('LapNumber')

def lap_times_plot(data,drivers):
    # figs = []
    height = 300
    fig = make_subplots(rows=len(drivers),cols=1,shared_xaxes=True,vertical_spacing=0.1,row_heights=[height]*len(drivers),subplot_titles=drivers)
    coms = []
    for i,d in enumerate(drivers):
        df = data.laps.pick_driver(d)
        df['LapTime'] = df['LapTime'] + dt.datetime(1970, 1, 1, 0, 0, 0,
                                                    0)  
        stints = df[["Driver", "Stint", "Compound", "LapNumber"]].copy()
        stints = stints.groupby(["Driver", "Stint", "Compound"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        pit_stops = data.laps.pick_driver(d).pick_box_laps()
        pit_stops = pit_stops[~pit_stops.PitInTime.isna()].LapNumber.to_list()
        for com in df.Compound.unique():
            subdata = df[df['Compound']==com]
            fig.add_trace(
                go.Scatter(x=subdata['LapNumber'].to_numpy(), y=subdata['LapTime'].to_numpy(),mode="markers",marker=dict(color=fastf1.plotting.COMPOUND_COLORS[com]),
                           showlegend=False,name=d),
                row=i+1,
                col=1,
            )
            coms.append(com)
        for p in pit_stops:
            fig.add_vline(x=p, line_width=3, line_dash='dash', line_color=fastf1.plotting.driver_color(d), row=i+1, col=1)
        rcm = data.race_control_messages
        ymin = data.laps.pick_driver(d).pick_fastest()['LapTime'] + dt.datetime(1970, 1, 1, 0, 0, 0, 0)
        ymax = (data.laps.pick_driver(d)['LapTime'].sort_values().iloc[-2:-1] + dt.datetime(1970, 1, 1, 0, 0, 0, 0)).iloc[0]
        if 'YELLOW' in data.race_control_messages.Flag.unique():
            yellow_laps = rcm[(rcm.Flag == 'YELLOW') & (rcm.Scope == 'Track')]['Lap'].unique()
            for l in yellow_laps:
                fig.add_annotation(
                    x=l,  # x-coordinate of the annotation
                    y=ymin,
                    # y-coordinate of the annotation
                    text="&#9888;",  # text to display
                    showarrow=False,
                    hovertext="YELLOW FLAG",
                    row = i+1,
                    col = 1
                )
        if 'RED' in data.race_control_messages.Flag.unique():
            yellow_laps = rcm[(rcm.Flag == 'RED') & (rcm.Scope == 'Track')]['Lap'].unique()
            for l in yellow_laps:
                fig.add_annotation(
                    x=l,  # x-coordinate of the annotation
                    y=ymin,
                    # y-coordinate of the annotation
                    text="&#128681;",  # text to display
                    hovertext="YELLOW FLAG",
                    showarrow=False,
                    row = i+1,
                    col = 1
                )
        if 'SafetyCar' in rcm.Category.unique():
            safety_car_laps = rcm[rcm.Category.apply(lambda x: 'SafetyCar' in x)]
            for _,row in safety_car_laps.iterrows():
                if row['Status'] == 'DEPLOYED':
                    start = row['Lap']
                    end = start
                    virtual_flag = 'VIRTUAL' in row['Message']
                if row['Status'] in ('ENDING','IN THIS LAP'):
                    end = row['Lap']
                    fig.add_shape(
                        type="rect",
                        x0 = start-0.5,
                        x1 = end+0.5,
                        y0 = ymin,
                        y1 = ymax + dt.timedelta(seconds=5),
                        fillcolor="yellow",
                        opacity=0.4,
                        row=i+1,
                        col=1
                    )
                    fig.add_annotation(
                        x=(start+end)/2,
                        y=ymin+dt.timedelta(seconds=10),
                        text=f"{'V' if virtual_flag else ''}SC",
                        showarrow=False,
                        row=i+1,
                        col=1,
                        textangle=0
                    )        
        fig.update_yaxes(
            tickformat='%M:%S.%f',
            row=i+1,
            col=1
        )
    for com in set(coms):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=com,
            marker=dict(color=fastf1.plotting.COMPOUND_COLORS[com])),
            row=1,col=1
        )
        
    fig.update_layout(
        xaxis_range=[0, data.laps.LapNumber.max()+1],
        height=height*len(drivers),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_speed_segments(data, drivers, fastest_lap=True):
    circuit_info = data.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi
    lap = data.laps.pick_fastest()
    pos = lap.get_telemetry()
    dist_segments = circuit_info.marshal_sectors.Distance.to_list()
    if dist_segments[0] > dist_segments[-1]:
        first = 0
        last = dist_segments[0]
        dist_segments[0] = first
        dist_segments.append(last)
    pos['dist_segments'] = pd.cut(pos.Distance, bins=dist_segments)
    driver_df = pd.Series()
    lap_time = {}
    for d in drivers:
        lap_time[d] = convert_timedelta_to_time(
            data.laps.pick_driver(d).pick_quicklaps().sort_values('LapTime').iloc[0]['LapTime'])
        if fastest_lap:
            temp_df = data.laps.pick_driver(d).pick_fastest().get_telemetry()
        else:
            temp_df = data.laps.pick_driver(d).pick_quicklaps().get_telemetry()
        temp_df['dist_segments'] = pd.cut(temp_df.Distance, bins=dist_segments)
        temp_df = temp_df.groupby('dist_segments')['Speed'].mean().reset_index()
        temp_df['Driver'] = d
        driver_df = pd.concat([driver_df, temp_df])

    max_speeds = driver_df.groupby('dist_segments')['Speed'].max().reset_index()
    driver_df = pd.merge(driver_df, max_speeds, on=['dist_segments', 'Speed'], how='inner')
    driver_df = driver_df.sort_values('dist_segments')
    pos = pd.merge(driver_df, pos, on=['dist_segments'], how='right')
    track = pos.loc[:, ('X', 'Y')].to_numpy()
    pos['X_unrotated'] = pos['X'].copy()
    pos['Y_unrotated'] = pos['Y'].copy()
    rotated_track = rotate(track, angle=track_angle)
    pos['X'] = rotated_track[:, 0]
    pos['Y'] = rotated_track[:, 1]
    pos['Driver_Colors'] = pos['Driver'].map(lambda x: fastf1.plotting.driver_color(x) if isinstance(x, str) else None)
    fig = go.Figure()
    start_pos = []
    prev_pos = []
    plot_pos = pd.DataFrame()
    for d in driver_df.Driver.unique():
        dom_segments = driver_df[driver_df['Driver'] == d]['dist_segments'].unique()
        sub_pos = pos[pos['dist_segments'].isin(dom_segments)]
        sub_pos['Driver'] = d
        plot_pos = pd.concat([plot_pos, sub_pos])

    plot_pos = plot_pos.sort_values('Distance')
    for count_plots, ds in enumerate(plot_pos.dist_segments.unique()):
        plot_pos1 = plot_pos[plot_pos['dist_segments'] == ds].sort_values('Distance')
        if count_plots == 0:
            first_segment_begin = [plot_pos1['X_unrotated'].iloc[0], plot_pos1['Y_unrotated'].iloc[0]]
            first_segment_end = [plot_pos1['X_unrotated'].iloc[-1], plot_pos1['Y_unrotated'].iloc[-1]]
            first_segment_angle = np.arctan(
                (first_segment_end[1] - first_segment_begin[1]) / (first_segment_end[0] - first_segment_begin[0]))
            start_pos = [plot_pos1['X'].iloc[0], plot_pos['Y'].iloc[0]]
        else:
            plot_pos1['X'].iloc[0] = prev_pos[0]
            plot_pos1['Y'].iloc[0] = prev_pos[1]
        fig.add_trace(
            go.Scatter(x=plot_pos1['X'], y=plot_pos1['Y'], mode='lines',
                       line=dict(color=fastf1.plotting.driver_color(plot_pos1['Driver'].iloc[0]), width=10),
                       hoverinfo='skip',showlegend=False)
        )
        prev_pos = [plot_pos1['X'].iloc[-1], plot_pos1['Y'].iloc[-1]]
        last_driver = plot_pos1['Driver'].iloc[0]

    fig.add_trace(
        go.Scatter(x=[prev_pos[0], start_pos[0]], y=[prev_pos[1], start_pos[1]], mode='lines',
                   line=dict(color=fastf1.plotting.driver_color(last_driver), width=10), hoverinfo='skip',showlegend=False)
    )

    if fastest_lap:
        title = 'Fastest Lap Comparison'
    else:
        title = 'Throughout the Session'
    
    x_range = pos['X'].max()+500 - (pos['X'].min()-500)
    y_range = pos['Y'].max()+500 - (pos['Y'].min()-500)
    h_by_w = y_range / x_range
    fig.update_layout(title=title, xaxis=dict(visible=False, range = [pos[f'X'].min()-500,pos[f'X'].max()+500]),
                      yaxis=dict(visible=False, range = [pos['Y'].min()-500,pos['Y'].max()+500]),
                      width=1000, height=1000*h_by_w,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    offset_vector = [300, 0]
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        offset_angle = corner['Angle'] / 180 * np.pi
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        fig.add_trace(
            go.Scatter(x=[text_x], y=[text_y], mode='text', text=txt, textposition='middle center', hoverinfo='skip',
                       textfont=dict(size=20))
        )
        fig['data'][-1]['showlegend'] = False

    # Add checkered flag
    offset_angle = np.pi / 2 + first_segment_angle
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
    text_x = first_segment_begin[0] + offset_x
    text_y = first_segment_begin[1] + offset_y
    text_x, text_y = rotate([text_x, text_y], angle=track_angle)
    emoji = '&#127937;'
    text = f"<span style='font-size:{20}px;'>{emoji}</span>"
    fig.add_trace(
        go.Scatter(x=[text_x], y=[text_y], mode='text', text=text, textposition='middle center', hoverinfo='skip',
                   textfont=dict(size=20),showlegend=False)
    )
    for d in drivers:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=d,
            line=dict(color=fastf1.plotting.driver_color(d)),
        ))
        fig.update_traces(dict(showlegend=True), selector=({'name': d}))
    lap_time = pd.DataFrame(lap_time.values(), columns=['LapTime'], index=lap_time.keys()).sort_values(
        'LapTime').reset_index()
    lap_time.rename({'index': 'Driver'}, axis=1, inplace=True)
    kpi_dict = {}
    if fastest_lap:
        for i in range(min(len(lap_time), 5)):
            minutes = int(lap_time['LapTime'].iloc[i] // 60)
            seconds = int(lap_time['LapTime'].iloc[i] % 60)
            milli = int(str(lap_time['LapTime'].iloc[i]).split('.')[-1].strip())
            kpi_dict[lap_time['Driver'].iloc[i]] = f"{minutes:>02d}:{seconds:>02d}.{milli:<03d}"

    return fig, kpi_dict

def get_track_animation_data(data,driver,lap_number):
    if lap_number == None:
        lap = data.laps.pick_driver(driver).pick_fastest().LapTime.iloc[0]
        df = data.laps.pick_driver(driver).pick_fastest().get_telemetry()
    else:
        lap = data.laps.pick_driver(driver).pick_lap(lap_number).LapTime.iloc[0]
        df = data.laps.pick_driver(driver).pick_lap(lap_number).get_telemetry() 

    return lap, df

def track_animation(data, drivers, lap_numbers):
    circuit_info = data.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi
    laps = []
    dfs = {}
    for i,d in enumerate(drivers):
        lap, df = get_track_animation_data(data,d,lap_number=lap_numbers[d])
        laps.append(lap)
        dfs[d] = df

    t0 = dt.datetime(1970,1,1,0,0,0,0)
    t1 = np.min(laps)
    delta_t = dt.timedelta(seconds=1)
    ts = pd.DataFrame(np.arange(t0,t1,delta_t)).rename({0: 'Time'},axis=1)
    ts['Time'] = ts['Time'] - t0
    offset_vector = [100, 0]
    offset_x, offset_y = rotate(offset_vector, angle=track_angle)
    driver_pos = pd.DataFrame()
    for i,d in enumerate(drivers):
        track = dfs[d].loc[:, ('X', 'Y')].to_numpy()
        dfs[d]['X_unrotated'] = dfs[d]['X'].copy()
        dfs[d]['Y_unrotated'] = dfs[d]['Y'].copy()
        rotated_track = rotate(track, angle=track_angle)
        dfs[d]['X'] = rotated_track[:, 0]
        dfs[d]['Y'] = rotated_track[:, 1]

        interpolation = dfs[d]
        interpolation['X'] = interpolation['X'] + i*offset_x
        interpolation['Y'] = interpolation['Y'] + i*offset_y

        if i == 0: 
            first_segment_begin = [interpolation['X_unrotated'].iloc[0],interpolation['Y_unrotated'].iloc[0]]
            first_segment_angle = np.arctan((interpolation['Y_unrotated'].iloc[3]-interpolation['Y_unrotated'].iloc[0])/interpolation['X_unrotated'].iloc[3]-interpolation['Y_unrotated'].iloc[0])

        # Store in driver pos
        if len(driver_pos) == 0:
            driver_pos = interpolation.drop('Speed',axis=1) 
        else:
            driver_pos = pd.merge(driver_pos,interpolation.drop('Speed',axis=1),on=['Time'],how='outer')
        driver_pos.rename({'X':f'X_{d}','Y':f'Y_{d}'},axis=1,inplace=True)

    fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
    }
    fig_dict["layout"]["xaxis"] = {"visible": False, "range": [np.min([driver_pos[f'X_{d}'].min() for d in drivers])-500,np.max([driver_pos[f'X_{d}'].max() for d in drivers])+500], "title": ""}
    fig_dict["layout"]["yaxis"] = {"visible": False, "range": [np.min([driver_pos[f'Y_{d}'].min() for d in drivers])-500,np.max([driver_pos[f'Y_{d}'].max() for d in drivers])+500], "title": ""}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 60, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 60,
                                                                        "easing": "linear-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 60, "easing": "linear-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    # make data
    data_dict = {
        "x": list(driver_pos[f"X_{drivers[0]}"]),
        "y": list(driver_pos[f"Y_{drivers[0]}"]),
        "mode": "lines",
        "line": {"width":20, "color":'rgba(128,128,128,0.1)'},
        "name": "Track",
        "showlegend": False
    }
    fig_dict["data"].append(data_dict)
    
    corners_data = []
    # Corners
    offset_vector = [300, 0]
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        offset_angle = corner['Angle'] / 180 * np.pi
        
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        corners_data.append(dict(x=[text_x], y=[text_y], mode='text', text=txt, textposition='middle center', hoverinfo='skip',
                       textfont=dict(size=20),showlegend=False))
    fig_dict["data"] = fig_dict["data"] + corners_data
    # make frames
    times = list(np.arange(t0,t1,delta_t)) 
    times.append(t1+t0)
    for time in times: 
        time = pd.to_datetime(time) - pd.to_datetime(t0)
        minutes = int(time.seconds // 60)
        seconds = int(time.seconds % 60)
        milli = int(dt.datetime.strftime(time+t0,"%M:%S.%f").split('.')[-1].strip()[:3])
        frame = {"data": [], "name": f"{minutes:>02d}:{seconds:>02d}:{milli:>03d}"}
        for d in drivers:
            data = driver_pos[driver_pos.Time <= time][["Time",f"X_{d}",f"Y_{d}"]]
            data_dict = {
                "x": list(driver_pos[f"X_{drivers[0]}"]),
                "y": list(driver_pos[f"Y_{drivers[0]}"]),
                "mode": "lines",
                "line": {"width":20, "color":'rgba(128,128,128,0.1)'},
                "name": "Track",
                "showlegend": False
            }
            frame["data"].append(data_dict)
            data_dict = {
                "x": [data.sort_values('Time',ascending=False).dropna()[f"X_{d}"].iloc[0]],
                "y": [data.sort_values('Time',ascending=False).dropna()[f"Y_{d}"].iloc[0]],
                "mode": "markers",
                "marker": {
                    "size": 10,
                    "color":fastf1.plotting.driver_color(d)
                },
                "name": d
            }
            frame["data"].append(data_dict)
        fig_dict["frames"].append(frame)

        # Update slider
        slider_step = {"args": [
            [f"{minutes:>02d}:{seconds:>02d}:{milli:>03d}"],
            {"frame": {"duration": 60, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 60}}
            ],
            "label": f"{minutes:>02d}:{seconds:>02d}:{milli:>03d}",
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)
    fig.add_traces(
        corners_data
    )
    
    
    #Checkered flag
    offset_angle = np.pi / 2 + first_segment_angle
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
    text_x = first_segment_begin[0] + offset_x
    text_y = first_segment_begin[1] + offset_y
    text_x, text_y = rotate([text_x, text_y], angle=track_angle)
    emoji = '&#127937;'
    text = f"<span style='font-size:{20}px;'>{emoji}</span>"
    fig.add_trace(
        go.Scatter(x=[text_x], y=[text_y], mode='text', text=text, textposition='middle center', hoverinfo='skip',
                   textfont=dict(size=20),showlegend=False)
    )
    x_range = np.max([driver_pos[f'X_{d}'].max() for d in drivers])+500 - (np.min([driver_pos[f'X_{d}'].min() for d in drivers])-500)
    y_range = np.max([driver_pos[f'Y_{d}'].max() for d in drivers])+500 - (np.min([driver_pos[f'Y_{d}'].min() for d in drivers])-500)
    h_by_w = y_range / x_range
    fig.update_layout(
        width=1000, 
        height=1000*h_by_w,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    
    
    # Legend 
    for d in drivers:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=d,
            marker=dict(color=fastf1.plotting.driver_color(d), size=10),
        ))
        fig.update_traces(dict(showlegend=True), selector=({'name': d}))

    return fig

def telemetry(lap_numbers,data,drivers):
    track_animation_fig = track_animation(data,drivers,lap_numbers=lap_numbers)
    telemetry_fig, kpi_dict = basic_plots(data,drivers,lap_numbers=lap_numbers)

    return track_animation_fig, telemetry_fig, kpi_dict
