import argparse
import logging
import sys
import os
import shutil
import inspect
from traceback import print_exc
from collections import namedtuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"

class HTMLBuilder(object):
    
    def __init__(self,
                 output_dn:str,
                 scenario_name:str):
        '''
        Constructor
        
        Arguments
        ---------
        output_dn:str
            output directory name
        scenario_name:str
            name of scenario
        '''
        self.logger = logging.getLogger("HTMLBuilder")
        
        
        self.own_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
        
        self.output_dn = output_dn
        
        self.header='''
<!DOCTYPE html>
<html>
<head>
<title>ASAPy for {0}</title>

<link href="css/accordion.css" rel="stylesheet" />
<link href="css/table.css" rel="stylesheet" />
<link href="css/lightbox.min.css" rel="stylesheet" />
<link href="css/help-tip.css" rel="stylesheet" />

</head>
<body>
<script src="js/lightbox-plus-jquery.min.js"></script>      
<h1>{0}</h1>  
        '''.format(scenario_name)
        
        self.footer = '''
        
Powered by <a href="http://www.coseal.net">COSEAL</a> and <a href="http://www.ml4aad.org">ML4AAD</a> 
</body>
<script>
var acc = document.getElementsByClassName("accordion");
var i;

for (i = 0; i < acc.length; i++) {
    acc[i].onclick = function(){
        this.classList.toggle("active");
        this.nextElementSibling.classList.toggle("show");
  }
}
</script>        
</html>        
        '''
        
        
    def generate_html(self, data_dict:dict):
        '''

            Arguments
            ---------
            data_dict : OrderedDict
                {"top1" : {
                            "tooltip": str|None,
                            "subtop1: {  # generates a further bottom if it is dictionary
                                    "tooltip": str|None,
                                    ...
                                    }
                            "table": str|None (html table)
                            "figure" : str | None (file name)
                            }
                "top2: { ... }
        '''
        html = ""
        html += self.header
        
        for k,v in data_dict.items():
            html = self.add_layer(html_str=html, layer_name=k, data_dict=v)
        html += self.footer
        
        with open(os.path.join(self.output_dn, "report.html"), "w") as fp:
            fp.write(html)
        
        try: 
            if not os.path.isdir(os.path.join(self.output_dn,"css")):
                shutil.copytree(os.path.join(self.own_folder, "web_files", "css"), os.path.join(self.output_dn,"css"))
        except OSError:
            print_exc()
        try:
            if not os.path.isdir(os.path.join(self.output_dn,"images")): 
                shutil.copytree(os.path.join(self.own_folder, "web_files", "images"), os.path.join(self.output_dn,"images"))
        except OSError:
            print_exc()
        try: 
            if not os.path.isdir(os.path.join(self.output_dn,"js")):
                shutil.copytree(os.path.join(self.own_folder, "web_files", "js"), os.path.join(self.output_dn,"js"))
        except OSError:
            print_exc()
            
        
    def add_layer(self, html_str:str, layer_name, data_dict:dict):
        ''' 
        add a further layer of top data_dict keys
        '''
        tooltip = ""
        if data_dict.get("tooltip"):
            tooltip = "<div class=\"help-tip\"><p>{}</p></div>".format(data_dict.get("tooltip"))
        html_str += "<button class=\"accordion\">{0} {1}</button>\n".format(layer_name,tooltip)
        html_str += "<div class=\"panel\">\n"
        for k, v in data_dict.items():
            if isinstance(v, dict):
                html_str = self.add_layer(html_str, k, v)
            elif k == "figure":
                html_str +="<div align=\"center\">\n"
                html_str +="<a href=\"{0}\" data-lightbox=\"{0}\" data-title=\"{0}\"><img src=\"{0}\" alt=\"Plot\" width=\"600px\"></a>\n".format(v)
                html_str +="</div>\n"
            elif k == "table":
                html_str += "<div align=\"center\">\n{}\n</div>\n".format(v)
            elif k == "tooltip":
                pass #TODO
        
        html_str += "</div>"
        return html_str