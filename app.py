import pandas as pd
import numpy as np

import dash
from dash import Dash, html, dcc
from dash import dash_table
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv("assets/goodreads_library_export.csv")

df["Date Added"] = pd.to_datetime(df["Date Added"], errors="coerce")

# filter for 'Date Added' in '2025' and 'Exclusive Shelf' in 'read' or 'currently-reading'
mask_2025 = df["Date Added"].dt.year == 2025
mask_shelf = df["Exclusive Shelf"].isin(["read", "currently-reading"])
filtered = df[mask_2025 & mask_shelf].copy()
min_pages = int(filtered["Number of Pages"].min())
max_pages = int(filtered["Number of Pages"].max())

# treat 0 as "Unrated"
filtered.loc[filtered["My Rating"] == 0, "My Rating"] = np.nan

# derive "Month Read" from 'Date Added'
filtered["Month Read"] = filtered["Date Added"].dt.to_period("M").astype(str)

# Monthly pages read totals
month_totals = (
    filtered
    .groupby("Month Read", as_index=False)["Number of Pages"]
    .sum()
    .rename(columns={"Number of Pages": "month_pages"})
)

# get first date in each month to sort chronologically
month_first_date = (
    filtered
    .groupby("Month Read")["Date Added"]
    .min()
    .reset_index()
    .rename(columns={"Date Added": "first_date"})
)

# merge first date into month totals and sort by time
month_totals = month_totals.merge(month_first_date, on="Month Read")
month_totals = month_totals.sort_values("first_date").reset_index(drop=True)

# total pages across all months (for converting to proportions)
grand_total_pages = month_totals["month_pages"].sum()
# proportion of total pages per month (column width of each Marimekko bar)
month_totals["width_prop"] = month_totals["month_pages"] / grand_total_pages

# x positions (0–1) for Marimekko columns
month_totals["x_start"] = month_totals["width_prop"].cumsum().shift(fill_value=0)
month_totals["x_end"] = month_totals["x_start"] + month_totals["width_prop"]
month_totals["x_center"] = (month_totals["x_start"] + month_totals["x_end"]) / 2

# rating label and aggregation
def rating_label_from_value(val):
    if pd.isna(val):
        return "Unrated"
    try:
        val = float(val)
        if val.is_integer():
            return f"{int(val)}⭐"
        return f"{val:.1f}⭐"
    except Exception:
        return "Unrated"

# add rating label column for legend / hover
filtered["Rating Label"] = filtered["My Rating"].apply(rating_label_from_value)

# collect book list per (Month, Rating Label)
def summarize_books(group):
    pairs = list(zip(group["Title"], group["Author"]))
    # one book per line in the tooltip
    return "<br>".join(f"{t} ({a})" for t, a in pairs)

# aggregate book titles/authors for each month-rating segment
agg_books = (
    filtered
    .groupby(["Month Read", "Rating Label"])
    .apply(summarize_books)
    .reset_index()
    .rename(columns={0: "Books"})
)

# aggregate total pages per month-rating segment (for area/height)
agg_pages = (
    filtered
    .groupby(["Month Read", "Rating Label"], as_index=False)["Number of Pages"]
    .sum()
    .rename(columns={"Number of Pages": "pages"})
)

# join aggregated pages and book lists
agg = agg_pages.merge(agg_books, on=["Month Read", "Rating Label"], how="left")

# join month-level stats (total pages, x positions) into segment data
agg = agg.merge(
    month_totals[["Month Read", "month_pages", "x_center", "width_prop"]],
    on="Month Read",
    how="left"
)

# segment height (stacks to 1.00 per month)
agg["height"] = agg["pages"] / agg["month_pages"]

# color scale (RdPu) for rating colors
rdpu = px.colors.sequential.RdPu

rating_order = ["Unrated", "1⭐", "2⭐", "3⭐", "4⭐", "5⭐"]

# map each rating label to a RdPu seg
rating_colors = {
    "Unrated": rdpu[1],
    "1⭐": rdpu[2],
    "2⭐": rdpu[3],
    "3⭐": rdpu[5],
    "4⭐": rdpu[7],
    "5⭐": rdpu[-1],
}

fig = go.Figure()

legend_labels = ["Unrated", "1⭐", "2⭐", "3⭐", "4⭐", "5⭐"]

# add traces in order of rating
for label in legend_labels:
    # subset to all month segments with this rating label
    sub = agg[agg["Rating Label"] == label].copy()

    if sub.empty:
        # if no data for this rating, add dummy bar so it still appears in legend
        fig.add_bar(
            x=[None],
            y=[None],
            width=[0],
            name=label,
            marker_color=rating_colors[label],
            showlegend=True,
            hoverinfo="skip",
        )
    else:
        x = sub["x_center"]
        y = sub["height"]
        width = sub["width_prop"]

        # hoverdata carries extra fields into hovertemplate
        hoverdata = np.stack(
            [
                sub["Month Read"],   
                sub["pages"],          
                sub["month_pages"],   
                sub["Books"],      
                sub["Rating Label"], 
            ],
            axis=-1,
        )

        # add bar for this rating (one bar per month segment)
        fig.add_bar(
            x=x,
            y=y,
            width=width,
            name=label,
            marker_color=rating_colors[label],
            customdata=hoverdata,
            hovertemplate=(
                "Month: %{customdata[0]}<br>"
                "Rating: %{customdata[4]}<br>"
                "Pages in this segment: %{customdata[1]}<br>"
                "Total pages this month: %{customdata[2]}<br><br>"
                "<b>Books:</b> %{customdata[3]}<extra></extra>"
            ),
        )

fig.update_layout(
    font=dict(
        family="Geologica, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    ),
    barmode="stack",
    title="How My 2025 Reading Time Is Distributed Across Months and Ratings",
    margin=dict(t=80, l=50, r=30, b=50),
    legend=dict(
        traceorder="reversed" # put 5 stars at the top
    ),
    xaxis_fixedrange=True,  
    yaxis_fixedrange=True, 
)

# x-axis ticks at month centers with month labels
fig.update_xaxes(
    title="Month in 2025 (wider columns = more pages read that month)",
    showgrid=False,
    zeroline=False,
    range=[0, 1],
    tickvals=month_totals["x_center"],
    ticktext=month_totals["Month Read"],
)

fig.update_yaxes(
    title="Share of pages in that month by rating (0–100%)",
    tickformat=".0%",
    range=[0, 1],
)

font_url = (
    "https://fonts.googleapis.com/css2?"
    "family=Funnel+Display:wght@300..800&"
    "family=Geologica:wght@100..900&"
    "family=Titillium+Web:wght@400;700&display=swap"
)

app = Dash(__name__, external_stylesheets=[font_url])

app.layout = html.Div(
    style={
        "margin": "40px",
        "fontFamily": "Geologica, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    children=[
        # Store to keep track of current click selection 
        dcc.Store(id="selection-store"),
        html.H1("What Did I Just Read? 2025 in Goodreads"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.5fr 3.5fr",
                "gap": "24px",
                "alignItems": "flex-start",
                "marginTop": "24px",
            },
            children=[
                # text blurb + table
                html.Div(
                    children=[
                        html.Div(
                            id="text-blurb-container",
                            style={
                                "border": "1px dashed #999",
                                "borderRadius": "8px",
                                "padding": "16px",
                                "marginBottom": "16px",
                                "minHeight": "120px",
                            },
                            children=[
                                html.H3("The Year of the Reading Competition"),
                                html.P(
                                    "This past January, a family member challenged me to a reading competition,"
                                    " to see how many books we can read in a year. Clearly, I was more motivated"
                                    " in January, at the start of the competition, than the rest of the school year."
                                    " I usually spend beach days in the summer reading, however this year I got very unlucky"
                                    " and picked up a book that was very long and that I was very uninterested in, so that took me a long time to read."
                                ),
                                html.P("This dashboard reveals I may have a skewed rating system for"
                                       " designating stars to a book review. Most of my ratings are between 4-5 stars, with"
                                       " the lowest being 3 stars (except for unrated books). Maybe I need to be more 'mean' in my reviews."),
                                html.P("It is also clear that I generally gravitate towards longer books (> 260 pages), which puts me at a clear disadvantage in this competition!"),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.H3("2025 Books (Read & Currently Reading)"),
                                dash_table.DataTable(
                                    id="books-table",
                                    data=filtered[
                                        [
                                            "Month Read",
                                            "Title",
                                            "Author",
                                            "My Rating",
                                            "Number of Pages",
                                        ]
                                    ].to_dict("records"),
                                    columns=[
                                        {"name": "Month Read", "id": "Month Read"},
                                        {"name": "Title", "id": "Title"},
                                        {"name": "Author", "id": "Author"},
                                        {"name": "My Rating", "id": "My Rating"},
                                        {"name": "Pages", "id": "Number of Pages"},
                                    ],
                                    page_size=10,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "4px",
                                        "fontFamily": "Geologica, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                                        "fontSize": "13px",
                                    },
                                    style_header={"fontWeight": "bold"},
                                ),
                            ]
                        ),
                    ]
                ),

                # Marimekko chart and slider
                html.Div(
                    children=[
                        html.Div( 
                            style={"marginBottom": "16px"},
                            children=[
                                html.Div(
                                    "Filter by number of pages in book",
                                    style={
                                        "fontWeight": "bold",
                                        "marginBottom": "6px",
                                        "fontSize": "13px",
                                    },
                                ),
                                dcc.RangeSlider(
                                    id="pages-slider",
                                    min=min_pages,
                                    max=max_pages,
                                    step=10,
                                    value=[min_pages, max_pages],
                                    marks={
                                        min_pages: str(min_pages),
                                        max_pages: str(max_pages),
                                    },
                                    allowCross=False,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="reading-marimekko",
                            figure=fig,
                            style={"height": "600px"},
                            config={
                                "scrollZoom": False,
                                "doubleClick": False,
                                "displayModeBar": False,
                            },
                        ),
                    ]
                ),

            ],
        ),
        html.Div(
            style={
                "marginTop": "32px",
                "fontSize": "12px",
                "color": "#555",
                "maxWidth": "900px",
            },
            children=[
                html.Hr(),
                html.Strong("How to read this chart:"),
                html.P(
                    "Each column represents one month in 2025. The width of a column "
                    "shows how many pages I read that month (wider = more pages). "
                    "Within each column, the colored blocks stack to 100% and show how "
                    "that month’s pages are split across my ratings (Unrated through 5⭐)."
                ),
                html.P(
                    "Use the slider above the chart to limit the view to books within a "
                    "certain page range. You can also click a colored block in the chart "
                    "to filter the table to just the books in that month-and-rating combo; "
                    "click the same block again to clear the filter."
                ),
            ],
        ),
    ],
)

# callback 1: update table when user clicks chart and/or moves page-range slider
@app.callback(
    Output("books-table", "data"),
    Output("selection-store", "data"),
    Input("reading-marimekko", "clickData"),
    Input("pages-slider", "value"),
    State("selection-store", "data"),
)
def update_table(clickData, page_range, stored_selection):
    low, high = page_range

    # base filter: page range only
    df_page = filtered[
        filtered["Number of Pages"].between(low, high, inclusive="both")
    ].copy()

    # ff no click or no selection, we just use page-range filtered table
    if clickData is None or "points" not in clickData or len(clickData["points"]) == 0:
        dff = df_page
        return dff[
            [
                "Month Read",
                "Title",
                "Author",
                "My Rating",
                "Number of Pages",
            ]
        ].to_dict("records"), None

    # get clicked month and rating
    point = clickData["points"][0]
    month = point["customdata"][0]        # Month Read
    rating_label = point["customdata"][4] # Rating Label

    # current selection dict for comparison with previous selection
    current_selection = {"month": month, "rating": rating_label}

    # Clicking same block clears month and rating filter
    if stored_selection == current_selection:
        dff = df_page
        return dff[
            [
                "Month Read",
                "Title",
                "Author",
                "My Rating",
                "Number of Pages",
            ]
        ].to_dict("records"), None

    # otherwise apply both filters: page range + (month, rating)
    dff = df_page[
        (df_page["Month Read"] == month) &
        (df_page["Rating Label"] == rating_label)
    ]

    # update stored selection to the newly clicked (month, rating)
    stored_selection = current_selection

    return dff[
        [
            "Month Read",
            "Title",
            "Author",
            "My Rating",
            "Number of Pages",
        ]
    ].to_dict("records"), stored_selection

# callback 2: rebuild Marimekko when page-range slider changes
@app.callback(
    Output("reading-marimekko", "figure"),
    Input("pages-slider", "value"),
)
def update_marimekko(page_range):
    low, high = page_range

    # filter by page range
    df_range = filtered[
        filtered["Number of Pages"].between(low, high, inclusive="both")
    ].copy()

    # if nothing in range, return an empty figure with a message
    if df_range.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"No books between {low} and {high} pages",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return empty_fig

    # monthly pages read totals within range
    month_totals = (
        df_range
        .groupby("Month Read", as_index=False)["Number of Pages"]
        .sum()
        .rename(columns={"Number of Pages": "month_pages"})
    )

    # sort months chronologically for chart
    month_first_date = (
        df_range
        .groupby("Month Read")["Date Added"]
        .min()
        .reset_index()
        .rename(columns={"Date Added": "first_date"})
    )

    # recompute monthly pages read totals within current page range
    month_totals = month_totals.merge(month_first_date, on="Month Read")

    # sort months chronologically for the x-axis
    month_totals = month_totals.sort_values("first_date").reset_index(drop=True)

    # total pages within range (used for width proportions)
    grand_total_pages = month_totals["month_pages"].sum()
    month_totals["width_prop"] = month_totals["month_pages"] / grand_total_pages

    # x positions (0–1) for Marimekko columns
    month_totals["x_start"] = month_totals["width_prop"].cumsum().shift(fill_value=0)
    month_totals["x_end"] = month_totals["x_start"] + month_totals["width_prop"]
    month_totals["x_center"] = (month_totals["x_start"] + month_totals["x_end"]) / 2

    # group books by month and rating for the tooltip strings
    agg_books = (
        df_range
        .groupby(["Month Read", "Rating Label"])
        .apply(summarize_books)
        .reset_index()
        .rename(columns={0: "Books"})
    )

    # aggregate pages per month-rating segment within the current page range
    agg_pages = (
        df_range
        .groupby(["Month Read", "Rating Label"], as_index=False)["Number of Pages"]
        .sum()
        .rename(columns={"Number of Pages": "pages"})
    )

    agg = agg_pages.merge(agg_books, on=["Month Read", "Rating Label"], how="left")
    agg = agg.merge(
        month_totals[["Month Read", "month_pages", "x_center", "width_prop"]],
        on="Month Read",
        how="left"
    )

    # recompute segment height = share of each month's pages for that rating
    agg["height"] = agg["pages"] / agg["month_pages"]

    rdpu = px.colors.sequential.RdPu
    rating_colors = {
        "Unrated": rdpu[1],
        "1⭐": rdpu[2],
        "2⭐": rdpu[3],
        "3⭐": rdpu[5],
        "4⭐": rdpu[7],
        "5⭐": rdpu[-1],
    }
    legend_labels = ["Unrated", "1⭐", "2⭐", "3⭐", "4⭐", "5⭐"]

    fig = go.Figure()

    # add bar trace for each rating, similar to initial construction
    for label in legend_labels:
        sub = agg[agg["Rating Label"] == label].copy()

        if sub.empty:
            fig.add_bar(
                x=[None],
                y=[None],
                width=[0],
                name=label,
                marker_color=rating_colors[label],
                showlegend=True,
                hoverinfo="skip",
            )
        else:
            x = sub["x_center"]
            y = sub["height"]
            width = sub["width_prop"]

            customdata = np.stack(
                [
                    sub["Month Read"],    # 0
                    sub["pages"],         # 1
                    sub["month_pages"],   # 2
                    sub["Books"],         # 3
                    sub["Rating Label"],  # 4
                ],
                axis=-1,
            )

            fig.add_bar(
                x=x,
                y=y,
                width=width,
                name=label,
                marker_color=rating_colors[label],
                customdata=customdata,
                hovertemplate=(
                    "Month: %{customdata[0]}<br>"
                    "Rating: %{customdata[4]}<br>"
                    "Pages in this segment: %{customdata[1]}<br>"
                    "Total pages this month: %{customdata[2]}<br><br>"
                    "<b>Books:</b><br>%{customdata[3]}<extra></extra>"
                ),
            )

    fig.update_xaxes(
        title="Month in 2025 (wider columns = more pages read that month)",
        showgrid=False,
        zeroline=False,
        range=[0, 1],
        tickvals=month_totals["x_center"],
        ticktext=month_totals["Month Read"],
        fixedrange=True,
    )

    fig.update_yaxes(
        title="Share of pages in that month by rating (0–100%)",
        tickformat=".0%",
        range=[0, 1],
        fixedrange=True,
    )

    fig.update_layout(
        font=dict(
        family="Geologica, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        size=13, 
        ),
        barmode="stack",
        title="How My 2025 Reading Time Is Distributed Across Months and Ratings",
        margin=dict(t=80, l=50, r=30, b=50),
        legend=dict(traceorder="reversed"),
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
