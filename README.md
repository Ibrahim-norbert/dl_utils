Replace in requirements.txt "data_loader" with data-loader as mentioned in 

Create requirements.txt with for windows with pip freeze | ForEach-Object { $_.Split('==')[0] } | Out-File -Encoding utf8 requirements.txt
