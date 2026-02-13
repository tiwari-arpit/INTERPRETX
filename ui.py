


import gradio as gr

def on_predict(input_json, model_choice, input_method, file_upload):
	# Placeholder: connect to backend later
	return ["", "", "", "", "", "", "", None]

with gr.Blocks(
	theme=gr.themes.Monochrome(
		primary_hue="orange",
		neutral_hue="slate",
		font=["Inter", "sans-serif"]
	),
	title="INTERPRETX – Explainable AI Platform",
	css="""
	.gradio-container { background: #18181b; }
	.input-card, .output-card {
		background: #232329;
		border-radius: 16px;
		box-shadow: 0 2px 12px 0 #111114;
		padding: 32px 24px 24px 24px;
		margin: 0 0 16px 0;
	}
	.gr-button { font-size: 1.1em; border-radius: 8px; font-weight:600; }
	.gr-textbox label, .gr-dropdown label { font-weight: 600; color: #f1f5f9; }
	.gr-markdown h1, .gr-markdown h2 { color: #fbbf24; }
	.gr-markdown, .gr-markdown div, .gr-markdown p { color: #e5e7eb; }
	.gr-tabitem { background: #18181b; }
	.gr-tabs { border-bottom: 1px solid #27272a; }
	"""
) as demo:
	gr.Markdown("""
		<div style='text-align:left; margin-bottom: 0.5em;'>
			<h1 style='font-weight:700; color:#fbbf24; margin-bottom:0;'>INTERPRETX</h1>
			<div style='font-size:1.1em; color:#e5e7eb;'>Decision Intelligence and Explainable AI Platform</div>
		</div>
	""")
	with gr.Row():
		# Left Panel: Input Section (Card Style)
		with gr.Column(scale=1, min_width=370, elem_classes=["input-card"]):
			gr.Markdown("""
				<h2 style='color:#fbbf24; font-weight:600; margin-bottom:0.5em;'>Input</h2>
			""")
			model_choice = gr.Dropdown(
				choices=[],
				label="Select Model",
				info="Model list will be loaded dynamically"
			)

			input_method = gr.Radio([
				"Manual Input",
				"File Upload"
			], value="Manual Input", label="Input Method", interactive=True)

			input_format = gr.Radio([
				"JSON",
				"CSV"
			], value="JSON", label="Input Format", interactive=True)

			input_text = gr.Textbox(
				label="Input Data",
				placeholder="Paste input JSON or CSV row here",
				lines=6,
				visible=True
			)
			file_upload = gr.File(label="Upload File", file_types=[".json", ".csv"], visible=False)
			file_preview = gr.Dataframe(label="File Preview", visible=False)

			def toggle_input(method):
				return gr.update(visible=method=="Manual Input"), gr.update(visible=method=="File Upload"), gr.update(visible=False)
			input_method.change(toggle_input, inputs=input_method, outputs=[input_text, file_upload, file_preview])

			def show_file_preview(file):
				import pandas as pd
				if file is None:
					return gr.update(visible=False), None
				try:
					ext = file.name.split('.')[-1].lower()
					if ext == 'csv':
						df = pd.read_csv(file.name)
					elif ext == 'json':
						df = pd.read_json(file.name)
					else:
						return gr.update(visible=True), [["Unsupported file type"]]
					return gr.update(visible=True), df.head(10)
				except Exception:
					return gr.update(visible=True), [["Error reading file"]]
			file_upload.change(show_file_preview, inputs=file_upload, outputs=[file_preview, file_preview])

			predict_btn = gr.Button("Predict", variant="primary", size="lg")
			interpret_btn = gr.Button("Interpret Decision", variant="secondary", size="lg")

		# Right Panel: Header/Nav with Train and Evaluation Tabs
		with gr.Column(scale=2, min_width=500, elem_classes=["output-card"]):
			with gr.Tabs(selected=0) as main_tabs:
				with gr.TabItem("Train"):
					gr.Markdown("""
						<h2 style='color:#fbbf24; font-weight:600; margin-bottom:0.5em;'>Train Model</h2>
						<div style='color:#e5e7eb;'>
							Training status and graphs will appear here after you start training from the left panel.
						</div>
					""")
					training_status = gr.Textbox(label="Training Status", interactive=False, placeholder="Status will appear here")
					training_graph = gr.Plot(label="Training Graph (Coming Soon)")
					model_data = gr.Dataframe(headers=["Feature", "Value"], label="Model Data", interactive=False)

				with gr.TabItem("Evaluation", id="evaluation_tab"):
					gr.Markdown("""
						<h2 style='color:#fbbf24; font-weight:600; margin-bottom:0.5em;'>Explanation & Evaluation</h2>
						<div style='color:#e5e7eb;'>
							Explanation details for the selected/trained AI model will appear here.
						</div>
					""")
					with gr.Tabs(selected=0):
						with gr.TabItem("Prediction"):
							prediction = gr.Textbox(label="Prediction", interactive=False, placeholder="Prediction will appear here")
							confidence = gr.Textbox(label="Confidence", interactive=False, placeholder="Confidence score")
							confidence_plot = gr.Plot(label="Confidence Gauge (Coming Soon)")
						with gr.TabItem("Counterfactuals"):
							counterfactual = gr.Textbox(label="Counterfactual Explanation", interactive=False, placeholder="Counterfactual reasoning")
						with gr.TabItem("Stability"):
							stability = gr.Textbox(label="Stability / Robustness", interactive=False, placeholder="Stability signal")
						with gr.TabItem("Uncertainty"):
							uncertainty = gr.Textbox(label="Uncertainty / Confidence", interactive=False, placeholder="Uncertainty signal")
						with gr.TabItem("Prototypes"):
							prototype = gr.Textbox(label="Prototype / Similarity", interactive=False, placeholder="Prototype/similarity explanation")
						with gr.TabItem("Fairness"):
							fairness = gr.Textbox(label="Fairness/Bias", interactive=False, placeholder="Fairness and bias metrics")
						with gr.TabItem("Governance"):
							governance = gr.Textbox(label="Governance Decision", interactive=False, placeholder="Governance policy decision")

			# Button logic (placeholders)

			predict_btn.click(
				on_predict,
				inputs=[input_text, model_choice, input_method, input_format, file_upload],
				outputs=[prediction, confidence, counterfactual, stability, uncertainty, prototype, governance, confidence_plot]
			)

			# Interpret button can be wired similarly later

if __name__ == "__main__":
	demo.launch()
