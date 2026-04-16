# 🧠 TreeSkill - Improve Prompts With Structure

[![Download TreeSkill](https://img.shields.io/badge/Download-TreeSkill-blue?style=for-the-badge&logo=github)](https://github.com/abhijeeth2004/TreeSkill)

## 🚀 What TreeSkill Does

TreeSkill helps you improve system prompts for LLMs. It uses a step-by-step search process to test prompt changes, compare results, and keep the best version.

Use it when you want to:

- refine a prompt without starting over
- compare prompt versions side by side
- collect human feedback on model output
- export preference data for DPO training
- keep prompt changes organized in one place

TreeSkill is built for users who want a clear process for prompt work. It gives structure to prompt updates and helps you see what changed.

## 📥 Download and Run on Windows

Use this link to visit the project page and download TreeSkill:

[Open TreeSkill on GitHub](https://github.com/abhijeeth2004/TreeSkill)

### Steps to get started

1. Open the link above in your browser.
2. Click the green **Code** button on the GitHub page.
3. Choose **Download ZIP**.
4. Save the file to your computer.
5. Right-click the ZIP file and choose **Extract All**.
6. Open the extracted folder.
7. Look for a file named `README.md` or an app file such as `TreeSkill.exe` if one is included in the release.
8. If the project includes a setup file, open it and follow the on-screen steps.
9. If it runs from a folder, double-click the main launch file listed in the project files.

If you use Windows Defender SmartScreen, choose **More info** and then **Run anyway** only if you trust the source and the file comes from the link above.

## 🖥️ System Requirements

TreeSkill works best on a Windows PC with:

- Windows 10 or Windows 11
- 8 GB of RAM or more
- 2 GB of free disk space
- A stable internet connection
- A modern browser like Chrome, Edge, or Firefox

For smoother use with local model tools or larger prompt runs, 16 GB of RAM helps.

## 🛠️ What You Need Before You Start

Before you open TreeSkill, make sure you have:

- a Windows account with permission to install files
- a browser to open the GitHub page
- enough free space to extract the ZIP file
- access to the LLM or API you want to test

If the project uses a local runtime, it may also need:

- Python 3.10 or newer
- Node.js if the interface uses a web app
- a text editor for config files
- an API key for the model provider you use

## 📂 Main Parts of the Project

TreeSkill is built around a few core ideas:

- **Prompt tree**: a set of prompt versions arranged in branches
- **Textual gradient descent**: a way to improve prompts by making small edits and checking results
- **Beam search**: a search method that keeps the strongest prompt options
- **Human review**: a step where a person rates or labels outputs
- **Preference export**: data output that can be used for DPO training

These parts help you move from one prompt to the next in a controlled way.

## ⚙️ How to Use TreeSkill

### 1. Open the project

After you download and extract the project, open the main folder.

### 2. Start the app or script

Use the file or command listed in the project files. The exact launch method may vary by setup. Look for:

- `run.bat`
- `start.bat`
- `app.py`
- `main.py`
- an executable file

### 3. Load your prompt

Paste the system prompt you want to improve. You can use a prompt for:

- support chat
- content checks
- data labeling
- answer formatting
- task planning

### 4. Run prompt tests

TreeSkill sends your prompt through the evaluation flow and collects output from the model.

### 5. Review results

Check the output and mark which version works best. You can note:

- clarity
- correctness
- tone
- format
- task success

### 6. Keep the best branch

TreeSkill uses the review results to keep the strongest prompt path.

### 7. Export preference data

When you have enough examples, export the data for later training or analysis.

## 🧪 Example Use Cases

TreeSkill can help with many prompt tasks:

- improving a customer support prompt
- fixing a prompt that gives vague answers
- testing prompt changes for a data extraction task
- comparing two versions of a system prompt
- building labeled preference data for model tuning

If you work with LLMs, this gives you a cleaner way to manage prompt updates.

## 🔍 How the Prompt Review Flow Works

TreeSkill follows a simple loop:

1. Start with one prompt.
2. Make a small change.
3. Test the new version.
4. Compare results.
5. Keep the better result.
6. Repeat the process.

This helps you avoid random prompt edits. Each change has a reason, and each result has a record.

## 🧾 File and Folder Guide

You may see these files after download:

- `README.md` — setup and usage info
- `requirements.txt` — Python package list
- `config.json` — app settings
- `prompts/` — prompt files
- `data/` — saved results and exports
- `src/` — source files
- `scripts/` — helper scripts

If the folder names are different, look for files with similar names.

## 🔐 Safe Setup Tips

Use the GitHub link above as your source file. When you download and extract the project:

- keep the files in one folder
- do not rename files unless the instructions say to
- do not move parts of the project into different folders
- use the latest version from the project page

If the app asks for an API key, enter only the key from your own account.

## 📊 Output and Export

TreeSkill can create files for later review or training. Common output may include:

- prompt score files
- branch comparison data
- review logs
- preference pairs
- export files for DPO workflows

Save exports in a place you can find later, such as Documents or a project folder on your Desktop.

## 🧩 Common Problems

### The file does not open

- Check that the ZIP file finished downloading.
- Extract the ZIP before opening files inside it.
- Look for the main launch file in the extracted folder.

### Windows blocks the file

- Right-click the file.
- Open **Properties**.
- Check whether an **Unblock** box appears.
- If it does, select it and click **OK**.

### The app cannot find a model or API

- Check your API key
- Check your internet connection
- Confirm the model name in the settings file
- Make sure the service you use is active

### Results look the same each time

- Add a small change to the prompt
- Check your input data
- Make sure the model settings are not locked to one fixed output
- Review the evaluation rules

## 🧠 Best Results

Use short, clear prompt changes. Keep one goal per test. Write down what you changed and why. This makes it easier to see which edit helped.

Good habits:

- change one thing at a time
- keep test cases simple
- save each good branch
- review outputs by the same rules each time
- export data in regular intervals

## 🧰 Suggested Workflow

1. Pick one system prompt.
2. Define what “better” means.
3. Run a first set of tests.
4. Review the outputs.
5. Keep the best prompt branch.
6. Test again with a new change.
7. Export the final preference data.

This workflow keeps your prompt work organized.

## 📁 Windows Setup Checklist

- Download the project from GitHub
- Extract the ZIP file
- Open the project folder
- Find the main run file
- Install any required tools
- Add your API key if needed
- Start the app
- Load a prompt
- Run a test
- Review the result
- Save the best version

## 📝 Notes for Non-Technical Users

If a screen asks for a folder path, choose a folder you can find again. If it asks for a file, use the files inside the extracted TreeSkill folder. If you are unsure which button to click, look for words like:

- Open
- Run
- Start
- Save
- Export
- Download

These usually point to the next step

## 🔗 Download Again

[Visit the TreeSkill project page](https://github.com/abhijeeth2004/TreeSkill)

## 📌 Project Name

TreeSkill