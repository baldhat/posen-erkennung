import os

model = ""

if len(model) > 1:
    folder_name = model
    os.chdir("models")
    os.chdir(folder_name)
    os.system("del {0}.tar.gz")
    os.system('"C:\\Program Files\\7-Zip\\7z.exe" a -ttar {0} *.*'.format(folder_name))
    os.system('"C:\\Program Files\\7-Zip\\7z.exe" a -tgzip {0}.tar.gz {0}.tar'.format(folder_name))
    os.system('"move {0}.tar.gz ..'.format(folder_name))
    os.system('del {0}.tar'.format(folder_name))
    os.chdir("..")
else:
    os.chdir("models")
    for folder_name in os.listdir("."):
        if os.path.isdir(folder_name):
            os.chdir(folder_name)
            os.system("del {0}.tar.gz")
            os.system('"C:\\Program Files\\7-Zip\\7z.exe" a -ttar {0} *.*'.format(folder_name))
            os.system('"C:\\Program Files\\7-Zip\\7z.exe" a -tgzip {0}.tar.gz {0}.tar'.format(folder_name))
            os.system('"move {0}.tar.gz ..'.format(folder_name))
            os.system('del {0}.tar'.format(folder_name))
            os.chdir("..")




