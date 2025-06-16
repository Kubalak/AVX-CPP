# from cgitb import html
import pathlib, os, datetime


project = "AVX CPP"
author = "Kubalak"
copyright = '2024-%Y, Kubalak'
version = '0.8.1.2'

extensions = ["breathe", "sphinx.ext.viewcode", "sphinx.ext.todo"]

html_theme = "furo"

breathe_projects = {
    "AVX_CPP": "../docs/xml"
}

breathe_projects_source = {
    "AVX_CPP": ("..", ["src/types/"])
}

breathe_default_project = "AVX_CPP"

breathe_show_include = True

viewcode_follow_imported_members=True

todo_include_todos=True