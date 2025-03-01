import gradio as gr
from tqdm import tqdm
import time
import json
import numpy as np
import plotly.colors
from itertools import chain
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from perspectives.ood_failure import extract_ood_examples
from perspectives.adv_demo_failure import extract_adv_demo
from perspectives.ethics_failure import extract_ethic_examples
from perspectives.fairness_failure import extract_fairness_examples
import pandas as pd
import random

DEFAULT_PLOTLY_COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


def to_rgba(rgb, alpha=1):
    return 'rgba' + rgb[3:][:-1] + f', {alpha})'

EXAMPLE_CACHE = {}


PERSPECTIVES = [
    "Toxicity", "Stereotype Bias", "Adversarial Robustness", "Out-of-Distribution Robustness",
    "Robustness to Adversarial Demonstrations", "Privacy", "Machine Ethics", "Fairness"
]
PERSPECTIVES_LESS = [
    "Toxicity", "Adversarial Robustness", "Out-of-Distribution Robustness",
    "Robustness to Adversarial Demonstrations", "Privacy", "Machine Ethics", "Fairness"
]


MAIN_SCORES = {
    "gpt-3.5-turbo-0301": [
        47,  # Toxicity
        87,  # Bias
        (67.37 + 49.23 + 50.42 + 59.73) / 4,  # Adv
        73.58311416938508,  # OoD
        0.8128416017653167 * 100,  # Adv Demo
        100 - 29.87106667,  # Privacy
        86.38,  # Machine Ethics
        100 * (1 - 0.2243)  # Fairness
    ],
    "gpt-4-0314": [
        41,  # Toxicity
        77,  # Bias
        (78.18 + 55.64 + 58.99 + 63.34) / 4,  # Adv
        87.54700929561338,  # OoD
        0.7794299606265144 * 100,  # Adv Demo
        100 - 33.8863,  # Privacy
        76.60,  # Machine Ethics
        100 * (1 - 0.3633)  # Fairness
    ],
    "alpaca-native": [
        22,  # Toxicity
        43,  # Bias
        (61.53 + 46.01 + 31.75) / 3,  # Adv
        51.785353417708116,  # OoD
        0.3415288335064037 * 100,  # Adv Demo
        100 - 53.60593333,  # Privacy
        30.43,  # Machine Ethics
        100 * (1 - 0.0737)  # Fairness
    ],
    "vicuna-7b-v1.3": [
        28,  # Toxicity
        81,  # Bias
        (52.55 + 52.21 + 51.71) / 3,  # Adv
        59.099378173030225,  # OoD
        0.5798818449290412 * 100,  # Adv Demo
        100 - 27.0362,  # Privacy
        48.22, # Machine Ethics
        100 * (1 - 0.1447)  # Fairness
    ],
    "Llama-2-7b-chat-hf": [
        80,  # Toxicity
        97.6,  # Bias
        (70.06 + 43.11 + 39.87) / 3,  # Adv
        75.65278958829596,  # OoD
        0.5553782796815506 * 100,  # Adv Demo
        100 - 2.605133333,  # Privacy
        40.58,  # Machine Ethics
        100  # Fairness
    ],
    "mpt-7b-chat": [
        40,  # Toxicity
        84.6,  # Bias
        (71.73 + 48.37 + 18.50) / 3,  # Adv
        64.26350715713153,  # OoD
        0.5825403080650745 * 100,  # Adv Demo
        100 - 21.07083333,  # Privacy
        26.11,  # Machine Ethics
        100 - 0  # Fairness
    ],
    "falcon-7b-instruct": [
        39,  # Toxicity
        87,  # Bias
        (73.92 + 41.58 + 16.44) / 3,  # Adv
        51.4498348176422,  # OoD
        0.33947969885773627 * 100,  # Adv Demo
        100 - 29.73776667,  # Privacy
        50.28,  # Machine Ethics
        100 - 0  # Fairness
    ],
    "RedPajama-INCITE-7B-Instruct": [
        18,
        73,  # Bias
        (66.02 + 48.22 + 20.20) / 3,  # Adv
        54.21313771953284,  # OoD
        0.5850598823122187 * 100,
        100 - 23.36082,  # Privacy
        27.49,  # Ethics
        100  # Fairness
    ]
}

ADV_TASKS = ["sst2", "qqp", "mnli"]
adv_results = {
    "hf/mosaicml/mpt-7b-chat": {"sst2": {"acc": 71.73}, "qqp": {"acc": 48.37}, "mnli": {"acc": 18.50}},
    "hf/togethercomputer/RedPajama-INCITE-7B-Instruct": {"sst2": {"acc": 66.02}, "qqp": {"acc": 48.22}, "mnli": {"acc": 20.2}},
    "hf/tiiuae/falcon-7b-instruct": {"sst2": {"acc": 73.92}, "qqp": {"acc": 41.58}, "mnli": {"acc": 16.44}},
    "hf/lmsys/vicuna-7b-v1.3": {"sst2": {"acc": 52.55}, "qqp": {"acc": 52.21}, "mnli": {"acc": 51.71}},
    "hf/chavinlo/alpaca-native": {"sst2": {"acc": 61.53}, "qqp": {"acc": 46.01}, "mnli": {"acc": 31.75}},
    "hf/meta-llama/Llama-2-7b-chat-hf": {"sst2": {"acc": 100 - 31.75}, "qqp": {"acc": 43.11}, "mnli": {"acc": 39.87}},
    "openai/gpt-3.5-turbo-0301": {"sst2": {"acc": 70.78}, "qqp": {"acc": 48.72}, "mnli": {"acc": 50.18}},
    "openai/gpt-4-0314": {"sst2": {"acc": 80.43}, "qqp": {"acc": 46.25}, "mnli": {"acc": 60.87}}
}

OOD_TASK = {"knowledge": ["qa_2020", "qa_2023"],
            "style": ["base", "shake_w", "augment", "shake_p0", "shake_p0.6", "bible_p0", "bible_p0.6", "romantic_p0",
                      "romantic_p0.6", "tweet_p0", "tweet_p0.6"]}

ADV_DEMO_TASKS = ["counterfactual", "spurious", "backdoor"]

TASK_SUBFIELDS = {"Toxicity":[
                    "nontoxic-benign-sys",
                    "toxic-benign-sys",
                    "toxic-gpt3.5-benign-sys",
                    "toxic-gpt4-benign-sys",
                    "nontoxic-adv-sys",
                    "toxic-adv-sys",
                    "toxic-gpt3.5-adv-sys",
                    "toxic-gpt4-adv-sys",
                     ],
                  "Stereotype Bias":["benign", "untargeted", "targeted"],
                  "Adversarial Robustness":["sst2", "qqp", "mnli"],
                  "Out-of-Distribution Robustness":[
                        "OoD Knowledge (Zero-shot)", "OoD Style (Zero-shot)", "OoD Knowledge (Few-shot)",
                        "OoD Style (Few-shot)",
                    ],
                  "Robustness to Adversarial Demonstrations":["counterfactual", "spurious", "backdoor"],
                  "Privacy":["enron", "PII", "understanding"],
                  "Machine Ethics":["jailbreaking prompts", "evasive sentence", "zero-shot benchmark", "few-shot benchmark"],
                  "Fairness":["zero-shot", "few-shot setting given unfair context", "few-shot setting given fair context"]}

TASK_CORRESPONDING_FIELDS = {"Out-of-Distribution Robustness":{"OoD Knowledge (Zero-shot)": "knowledge_zeroshot",
                              "OoD Style (Zero-shot)": "style_zeroshot",
                              "OoD Knowledge (Few-shot)": "knowledge_fewshot",
                              "OoD Style (Few-shot)": "style_fewshot"},
                              "Privacy":{"zero-shot": "zero-shot",
                              "few-shot setting given unfair context": "few-shot-1",
                              "few-shot setting given fair context": "few-shot-2"},
                              "Machine Ethics": {"jailbreaking prompts": "jailbreak",
                                "evasive sentence": "evasive"}
                              }
with open("./data/results/toxicity_results.json") as file:
    toxicity_results = json.load(file)

with open("./data/results/ood_results.json", "r") as file:
    ood_results = json.load(file)

with open("./data/results/adv_demo.json") as file:
    adv_demo_results = json.load(file)

with open("./data/results/fairness_results.json") as file:
    fairness_results = json.load(file)

with open("./data/results/ethics_results.json") as file:                                                                                                                                                                    
    ethics_results = json.load(file)

with open("./data/results/stereotype_results.json") as file:
    stereotype_results = json.load(file)

with open("./data/results/privacy_results.json") as file:
    privacy_results = json.load(file)

models_to_analyze = [
    "hf/mosaicml/mpt-7b-chat",
    "hf/togethercomputer/RedPajama-INCITE-7B-Instruct",
    "hf/tiiuae/falcon-7b-instruct",
    "hf/lmsys/vicuna-7b-v1.3",
    "hf/chavinlo/alpaca-native",
    "hf/meta-llama/Llama-2-7b-chat-hf",
    "openai/gpt-3.5-turbo-0301",
    "openai/gpt-4-0314"
]


def radar_plot(aggregate_keys, all_keys, results, thetas, title, metric, selected_models=None):
    # Extract performance values for each model across all benchmarks
    model_performance = {}
    # print("selected_models", selected_models)
    if selected_models is None:
        selected_models = models_to_analyze
    for model in selected_models:
        if model in results:
            benchmarks_data = results[model]
            if metric:
                model_performance[model] = [
                    np.nanmean([benchmarks_data[x][metric] if benchmarks_data[x][metric] is not None else np.nan
                                for x in all_keys if x.startswith(benchmark)]) for benchmark in aggregate_keys
                ]
            else:
                model_performance[model] = [
                    np.nanmean([list(benchmarks_data[x].values())[0] for x in all_keys if
                                x.startswith(benchmark)]) for benchmark in aggregate_keys
                ]
            if "counterfactual" in all_keys or "jailbreak" in all_keys or metric in ["Equalized Odds Difference", "Demographic Parity Difference", "emt", "category_overall_score"]:
                model_performance[model] = [x * 100 for x in model_performance[model]]
            if metric in ["asr", "Equalized Odds Difference", "Demographic Parity Difference", "emt", "brittleness"]:
                model_performance[model] = [100 - x for x in model_performance[model]]

    # Create radar chart with plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        row_heights=[1, 1],
        specs=[[{"type": "polar"}], [{"type": "table"}]]
    )

    for i, (model, performance) in enumerate(model_performance.items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]

        print(performance, aggregate_keys)
        fig.add_trace(
            go.Scatterpolar(
                r=performance + [performance[0]],
                theta=thetas + [thetas[0]],
                fill='toself',
                connectgaps=True,
                fillcolor=to_rgba(color, 0.1),
                name=model.split('/')[-1],  # Use the last part of the model name for clarity
            ),
            row=1, col=1
        )

    header_texts = ["Model"] + [x.replace("<br>", " ") for x in aggregate_keys]
    rows = [[x.split('/')[-1] for x in selected_models]] + [[round(score[i], 2) for score in [model_performance[x] for x in selected_models]] for i in range(len(aggregate_keys))]
    column_widths = [len(x) for x in header_texts]
    column_widths[0] *= 8 if "Toxicity" in title else 3

    fig.add_trace(
        go.Table(
            header=dict(values=header_texts, font=dict(size=15), align="left"),
            cells=dict(
                values=rows,
                align="left",
                font=dict(size=15),
                height=30
            ),
            columnwidth=column_widths
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=1000,
        legend=dict(font=dict(size=20), orientation="h", xanchor="center", x=0.5, y=0.55),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming accuracy is a percentage between 0 and 100
                tickfont=dict(size=12)
            ),
            angularaxis=dict(tickfont=dict(size=20), type="category")
        ),
        showlegend=True,
        title=f"{title}"
    )

    return fig


def main_radar_plot(perspectives, selected_models=None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        row_heights=[0.5, 0.5],
        specs=[[{"type": "polar"}], [{"type": "table"}]]
    )

    # perspectives_shift = (perspectives[4:] + perspectives[:4])  # [::-1
    perspectives_shift = perspectives
    model_scores = MAIN_SCORES
    if selected_models is not None:
        model_scores = {}
        for model in selected_models:
            select_name = os.path.basename(model)
            model_scores[select_name] = []
            for perspective in perspectives:
                score_idx = PERSPECTIVES.index(perspective)
                model_scores[select_name].append(MAIN_SCORES[select_name][score_idx])


    for i, (model_name, score) in enumerate(model_scores.items()):
        color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]

        # score_shifted = score[4:] + score[:4]
        score_shifted = score
        # print(score_shifted + [score_shifted[0]])
        fig.add_trace(
            go.Scatterpolar(
                r=score_shifted + [score_shifted[0]],
                theta=perspectives_shift + [perspectives_shift[0]],
                connectgaps=True,
                fill='toself',
                fillcolor=to_rgba(color, 0.1),
                name=model_name,  # Use the last part of the model name for clarity
            ),
            row=1, col=1
        )

    header_texts = ["Model"] + perspectives
    rows = [
        list(model_scores.keys()),  # Model Names
        *[[round(score[i], 2) for score in list(model_scores.values())] for i in range(len(perspectives))]
    ]
    column_widths = [10] + [5] * len(perspectives)

    fig.add_trace(
        go.Table(
            header=dict(values=header_texts, font=dict(size=15), align="left"),
            cells=dict(
                values=rows,
                align="left",
                font=dict(size=15),
                height=30,
            ),
            columnwidth=column_widths,
        ),
        row=2, col=1
    )


    fig.update_layout(
        height=1200,
        legend=dict(font=dict(size=20), orientation="h", xanchor="center", x=0.5, y=0.55),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming accuracy is a percentage between 0 and 100
                tickfont=dict(size=12)
            ),
            angularaxis=dict(tickfont=dict(size=20), type="category", rotation=5)
        ),
        showlegend=True,
        title=dict(text="DecodingTrust Scores (Higher is Better) of GPT Models"),
    )


    return fig


def breakdown_plot(selected_perspective, selected_models=None):
    if selected_models is None:
        selected_models = models_to_analyze
    if selected_perspective == "Main Figure":
        if selected_models is not None:
            selected_models = [os.path.basename(selected_model) for selected_model in selected_models]
        fig = main_radar_plot(PERSPECTIVES, selected_models)
    elif selected_perspective == "Adversarial Robustness":
        fig = radar_plot(
            ADV_TASKS,
            ADV_TASKS,
            adv_results,
            ADV_TASKS,
            selected_perspective,
            "acc",
            selected_models
        )
    elif selected_perspective == "Out-of-Distribution Robustness":
        # print({model: ood_results[model] for model in selected_models})
        fig = radar_plot(
            ["knowledge_zeroshot", "style_zeroshot", "knowledge_fewshot", "style_fewshot"],
            list(ood_results[models_to_analyze[0]].keys()),
            ood_results,
            [
                "OoD Knowledge (Zero-shot)", "OoD Style (Zero-shot)", "OoD Knowledge (Few-shot)",
                "OoD Style (Few-shot)",
            ],
            selected_perspective,
            "score",
            selected_models
        )
    elif selected_perspective == "Robustness to Adversarial Demonstrations":
        fig = radar_plot(
            ["counterfactual", "spurious", "backdoor"],
            ["counterfactual", "spurious", "backdoor"],
            adv_demo_results,
            ["counterfactual", "spurious", "backdoor"],
            selected_perspective,
            "",
            selected_models
        )
    elif selected_perspective == "Fairness":
        fig = radar_plot(
            ["zero-shot", "few-shot-1", "few-shot-2"],
            ["zero-shot", "few-shot-1", "few-shot-2"],
            fairness_results,
            ["zero-shot", "few-shot setting given unfair context", "few-shot setting given fair context"],
            selected_perspective,
            "Equalized Odds Difference",
            selected_models
        )
    elif selected_perspective == "Machine Ethics":
        fig = radar_plot(
            ["jailbreak", "evasive", "zero-shot benchmark", "few-shot benchmark"],
            ["jailbreak", "evasive", "zero-shot benchmark", "few-shot benchmark"],
            ethics_results,
            ["jailbreaking prompts", "evasive sentence", "zero-shot benchmark", "few-shot benchmark"],
            selected_perspective,
            "",
            selected_models
        )
    elif selected_perspective == "Privacy":
        fig = radar_plot(
            ["enron", "PII", "understanding"],
            ["enron", "PII", "understanding"],
            privacy_results,
            ["enron", "PII", "understanding"],
            selected_perspective,
            "asr",
            selected_models
        )
    elif selected_perspective == "Toxicity":
        fig = radar_plot(
            [
                 "nontoxic-benign-sys",
                 "toxic-benign-sys",
                 "toxic-gpt3.5-benign-sys",
                 "toxic-gpt4-benign-sys",
                 "nontoxic-adv-sys",
                 "toxic-adv-sys",
                 "toxic-gpt3.5-adv-sys",
                 "toxic-gpt4-adv-sys",
            ],
            [
                 "nontoxic-benign-sys",
                 "toxic-benign-sys",
                 "toxic-gpt3.5-benign-sys",
                 "toxic-gpt4-benign-sys",
                 "nontoxic-adv-sys",
                 "toxic-adv-sys",
                 "toxic-gpt3.5-adv-sys",
                 "toxic-gpt4-adv-sys",
            ],
            toxicity_results,
            [
                 "nontoxic-benign-sys",
                 "toxic-benign-sys",
                 "toxic-gpt3.5-benign-sys",
                 "toxic-gpt4-benign-sys",
                 "nontoxic-adv-sys",
                 "toxic-adv-sys",
                 "toxic-gpt3.5-adv-sys",
                 "toxic-gpt4-adv-sys",
            ],
            selected_perspective,
            "emt",
            selected_models
        )
    elif selected_perspective == "Stereotype Bias":
        fig = radar_plot(
            ["benign", "untargeted", "targeted"],
            ["benign", "untargeted", "targeted"],
            stereotype_results,
            ["benign", "untargeted", "targeted"],
            selected_perspective,
            "category_overall_score",
            selected_models
        )

    else:
        raise ValueError(f"Choose perspective from {PERSPECTIVES}!")
    return fig
def extract_failure(extract_fn, model, subfield, curr_examples):
    if model not in EXAMPLE_CACHE.keys():
        EXAMPLE_CACHE[model] = {}
    if subfield not in EXAMPLE_CACHE[model].keys():
        examples = extract_fn(model, subfield)
        random.shuffle(examples)
        EXAMPLE_CACHE[model][subfield] = examples
    examples = EXAMPLE_CACHE[model][subfield]
    if curr_examples + 10 > len(examples):
        examples = examples
        curr_button = gr.Button(value="More examples!", visible=False)
    else:
        examples = examples[:curr_examples+10]
        curr_button = gr.Button(value="More examples!", visible=True)
    if len(examples) > 0:
        df = pd.DataFrame.from_records(examples)
        df = gr.Dataframe(value=df, visible=True, row_count=[curr_examples + 10, "fixed"], col_count=[2, "fixed"], headers=["Query", "Outputs"])
    else:
        df = gr.Dataframe(value=pd.DataFrame.from_records([{"Query":"None", "Outputs": "None"}]), visible=True, row_count=[1, "fixed"], col_count=[2, "fixed"], headers=["Query", "Outputs"])
    return df, curr_button
    
def retrieve_fault_demo(model, categories, subfield, curr_examples):
    if categories == "Out-of-Distribution Robustness":
        df, curr_button = extract_failure(extract_ood_examples, model, subfield, curr_examples)
    elif categories == "Robustness to Adversarial Demonstrations":
        df, curr_button = extract_failure(extract_adv_demo, model, subfield, curr_examples)
    elif categories == "Machine Ethics":
        df, curr_button = extract_failure(extract_ethic_examples, model, subfield, curr_examples)
    elif categories == "Fairness":
        df, curr_button = extract_failure(extract_fairness_examples, model, subfield, curr_examples)
    else:
        df = gr.Dataframe(value=pd.DataFrame.from_records([{"Query":"Test Test Test", "Outputs": "Test Test Test"}] * 10), visible=True, row_count=[10, "fixed"], col_count=[2, "fixed"], headers=["Query", "Outputs"])
        curr_button = gr.Button(value="More examples!", visible=True)
    return df, curr_button

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column(visible=True) as model_col:
        model_selection = gr.Dropdown(
            choices=models_to_analyze,
            value="openai/gpt-4-0314",
            label="Select Model",
            allow_custom_value=True
        )
        perspectives = gr.CheckboxGroup(
            choices=PERSPECTIVES,
            label="Select Scenarios"
        )
        button = gr.Button(value="Submit")
    
    with gr.Column(visible=False) as output_col:
        curr_select = gr.Dropdown(
            choices=[],
            label="Select Scenario"
        )
        with gr.Column(visible=False) as output_col2:
            gr.Markdown("# Failure example")
            perspective_dropdown = gr.Dropdown()
            df_gr = gr.Dataframe(visible=False)
            add_button = gr.Button(value="More examples!", visible=False)
        gr.Markdown("# Overall statistics") 
        plot = gr.Plot()
        download_button = gr.Button()
  
    def radar(model, categories, categories_all):
        if len(categories) == 0 and model not in models_to_analyze: 
            pr=gr.Progress(track_tqdm=True)
            for category in pr.tqdm(categories_all, desc="Running selected scenarios"):
                for i in pr.tqdm(range(15), desc=f"Running {category}"):
                    time.sleep(0.1)

        categories_name = ["Main Figure"] + categories_all
        if len(categories) == 0 or categories == "Main Figure":
            fig = main_radar_plot(categories_all, [model])
            select = gr.Dropdown(choices=categories_name, value="Main Figure", label="Select Scenario")
            demo_col = gr.Column(visible=False)
            dropdown = gr.Dropdown(choices=[], label="Select Subscenario")
            download=gr.Button(link="/file=report.csv", value="Download Report", visible=True)
        else:
            fig = breakdown_plot(categories, [model])
            select = gr.Dropdown(choices=categories_name, value=categories, label="Select Scenario")
            demo_col = gr.Column(visible=True)
            dropdown = gr.Dropdown(choices=TASK_SUBFIELDS[categories], label="Select Subscenario")
            download=gr.Button(visible=False)
        return {plot: fig, output_col: gr.Column(visible=True), model_col: gr.Column(visible=False), curr_select: select, output_col2: demo_col, perspective_dropdown: dropdown, button:gr.Button(visible=False), model_selection:gr.Dropdown(visible=False), df_gr:gr.DataFrame(visible=False), add_button:gr.Button(visible=False), download_button:download}
    
    def retrieve_new_demo(model, categories, subfield):
        df, button = retrieve_fault_demo(model, categories, subfield, 0)
        return {df_gr: df, add_button: button}
    
    def retrieve_more_demo(model, categories, subfield, df):
        df, button = retrieve_fault_demo(model, categories, subfield, len(df))
        return {df_gr: df, add_button: button}
        
    gr.on(triggers=[button.click, curr_select.change], fn=radar, inputs=[model_selection, curr_select, perspectives], outputs=[plot, output_col, model_col, curr_select, output_col2, perspective_dropdown, button, model_selection, df_gr, add_button, download_button])
    perspective_dropdown.change(fn=retrieve_new_demo, inputs=[model_selection, curr_select, perspective_dropdown], outputs=[df_gr, add_button])
    add_button.click(fn=retrieve_more_demo, inputs=[model_selection, curr_select, perspective_dropdown, df_gr], outputs=[df_gr, add_button])

if __name__ == "__main__":
    demo.queue().launch(server_port=8089)


