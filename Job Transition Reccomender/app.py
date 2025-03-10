from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#these are just taken from the wordlist more can be added or taken away
career_level_mapping = {
    "junior": 0,
    "mid level": 1,
    "senior": 2,
    "lead": 3
}
job_category_mapping = {
    'developer': 0,
    'engineer': 1,
    'designer': 2,
    'analyst': 3,
    'tester': 4,
    'technical': 5,
    'technician': 6,
    'managerial': 7,
    'marketing': 8
}
hard_skills = [
  "configuration management",
    "sql",
    "html",
    "aws,amazon web services",
    "agile",
    "jira",
    "css",
    "c",
    "sqlserver,sql server,sqldatatools",
    "javascript",
    "indesign",
    ".net,dotnet,dotnetcore",
    "photoshop",
    "jenkins",
    "linux",
    "ruby",
    "http",
    "git",
    "react native",
    "csharp,c#",
    "selenium",
    "provisioning",
    "iot",
    "react",
    "scrum",
    "python",
    "cucumber,cucumberbdd",
    "puppet",
    "windows",
    "docker",
    "windows server",
    "version control",
    "illustrator",
    "java",
    "rails,ruby on rails,rubyonrails",
    "excel",
    "nginx",
    "less",
    "vmware*",
    "angular",
    "gradle",
    "tdd",
    "node.js,nodejs",
    "webpack",
    "stack",
    "specflow",
    "unix",
    "asp",
    "azure,azuredevops",
    "mongodb,mongo",
    "saas",
    "grunt",
    "gulp",
    "svn",
    "bootstrap",
    "sass",
    "drupal",
    "serverless",
    "symfony",
    "teamcity",
    "reactjs,react.js",
    "npm",
]

# Loading the model
model = joblib.load('model.pkl')


#render the html for the user
@app.route('/')
def index():
    return render_template('index.html', prediction="", hard_skills=hard_skills, binary_array=[])  

#GET user input from form and POST predicted answer with the form back.
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # Get data from the form
    job_category = request.form['jobCategory']
    selected_skills = request.form.get('hardSkills').split(',') 
    career_level = request.form['careerLevel']

    print("Selected Skills:", selected_skills)

    #map the career level and job category given back to the models integer format
    mapped_career_level = career_level_mapping.get(career_level.lower(), -1)
    mapped_job_category = job_category_mapping.get(job_category.lower(), -1)
    
    # Create binary array for hard skills
    binary_array = []
    for skill in hard_skills:
        matched = any(part.strip().lower() in [s.strip().lower() for s in selected_skills] for part in skill.split(','))
        binary_array.append(1 if matched else 0)

    print("Binary array:", binary_array)
    print("Career Level:", mapped_career_level)
    print("Job Category:", mapped_job_category)

    #define X for the model to read from the user
    input_features = [mapped_job_category, mapped_career_level] + binary_array
    #format X into an array for the model to read correctly
    input_features = np.array(input_features).reshape(1, -1)
    #model predicts based on X from user
    predicted_job_categories = model.predict(input_features)

    print("predicted job:", predicted_job_categories)    

    #return all values to site
    return render_template('index.html', predicted_job_categories=predicted_job_categories)


if __name__ == '__main__':
    app.run(debug=True)
