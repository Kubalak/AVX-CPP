import xml
import xml.etree
import xml.etree.ElementTree
from argparse import ArgumentParser

ALL_KINDS = ['public-type','private-attrib','public-static-attrib','private-static-func','public-func','friend']
EXPORTED = ['public-func','friend']

def getFunctionPrototype(section:xml.etree.ElementTree.Element):
    functions = section.findall("memberdef")   
     
    prototypes = [
        {
            'name': prototype.find('definition').text,
            'arguments': prototype.find('argsstring').text
        }
        for prototype in functions
    ]
    
    for prototype in prototypes:
        prototype['name'] = prototype['name'][prototype['name'].find('avx::'):]
        prototype['arguments'] = prototype['arguments'][0:prototype['arguments'].find(')') + 1]
    
    return prototypes


def parseXML(filename:str):
    tree = xml.etree.ElementTree.parse(filename)
    
    root  = tree.getroot()
    sections = [element for element in root.findall('./compounddef/sectiondef') if element.attrib['kind'] in EXPORTED]
    
    functions = []
    
    for section in sections:
        functions.extend(getFunctionPrototype(section))
    
    return functions


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("filename", help="Source filename", type=str)
    parser.add_argument("-output", "-o", help="Output filename", type=str, default="o.rst")
    
    namespace = parser.parse_args()
    
    functions = parseXML(namespace.filename)
    with open(namespace.output, "w") as output:
        for function in functions:
            output.write(f".. doxygenfunction:: {function['name']}{function['arguments'] if len(function['arguments']) > 2 else ''}")
            output.write("  :project: AVX_CPP\n")
    
    print(f"Results saved to {namespace.output}")
    
    