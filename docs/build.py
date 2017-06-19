import pdoc
s = pdoc.html('bayesian_bootstrap.bootstrap')
with open('bootstrap_documentation.html', 'w') as f:
    f.write(s)