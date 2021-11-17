# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
Application to extract coreference data from the CoNLL-2011 Shared Task dataset
and save it in the JSON format used by parc_extractor.py. Character indices are
adjusted to match those in PDTB v2.0 (minus the PDTB prefix).ß

Run the application once for each of the following directories in the CoNLL-2011 dataset:

    conll-2011/conll-2011/v2/data/train/data/english/annotations/nw/wsj/    (contains: 02 - 21)
    conll-2011/conll-2011_2/v2/data/dev/data/english/annotations/nw/wsj/    (contains: 00, 01, 22, 24)
    conll-2011/conll-2011_7/v2/data/test/data/english/annotations/nw/wsj/   (contains: 23)
"""

import xml.etree.ElementTree as ET
import argparse
import logging
import json
import re
import os

from citron.logger import logger

PDTB_PREFIX_LENGTH = 9  # The length of the prefix found in all PDTB files.


def main():
    parser = argparse.ArgumentParser(
        description="CoNLL-2011 Data Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v",
      action = "store_true",
      default = False,
      help = "Verbose mode"
    )
    parser.add_argument("--pdtb-path", 
      metavar = "pdtb_path",
      type = str,
      required=True,
      help = "Path to: pdtb_v2/data/raw/wsj"
    )
    parser.add_argument("--conll-path", 
      metavar = "conll_path",
      type = str,
      required=True,
      help = "Path to CoNLL-2011 data"
    )
    parser.add_argument("--output-path", 
      metavar = "output_path",
      type = str,
      required=True,
      help = "Path of output directory"
    )
    args = parser.parse_args()
    
    if args.v:
        logger.setLevel(logging.DEBUG)
    
    logger.info("CoNLL-2011 path: %s", args.conll_path)    
    logger.info("PDTB V2 path:    %s", args.pdtb_path)
    logger.info("Output path:     %s", args.output_path)
    
    stats = {"files": 0, "groups": 0, "coreferences": 0}
    conll_parser = ConllParser()
    process_directory(conll_parser, args.conll_path, args.pdtb_path, args.output_path, stats)
    
    logger.info("Processed files:    %s", stats["files"])
    logger.info("Coreference groups: %s", stats["groups"])
    logger.info("Coreferences:       %s", stats["coreferences"])


def process_directory(conll_parser, conll_directory, pdtb_directory, output_directory, stats):
    """
    Process a directory of CoNLL-2011 data.
    
    Args:
        conll_parser: An ConllParser object.
        conll_directory: The path of a CoNLL-2011 directory.
        pdtb_directory: The path of the equivalent PDTB v2 directory.
        output_directory: The path of the output directory.
        stats: A dict using to record statistics.
    """
    
    for conll_name in os.listdir(conll_directory):
        if conll_name.startswith("."):
            continue
        
        conll_path = os.path.join(conll_directory, conll_name)        
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        if os.path.isdir(conll_path):
            logger.debug("directoryName: %s", conll_path)
            pdtb_path = os.path.join(pdtb_directory, conll_name) 
            out_path = os.path.join(output_directory, conll_name)
            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            for filename in os.listdir(conll_path):
                if filename.endswith(".v2_gold_coref"):
                    conll_file = os.path.join(conll_path, filename)
                    pdtb_file  = os.path.join(pdtb_path, filename[: -14])
                    
                    if not os.path.exists(pdtb_file):
                        logger.debug("PDTB file not found: %s", pdtb_file)
                        continue
                    
                    output_file  = os.path.join(out_path, filename[: -14] + ".json")
                    process_file(conll_parser, conll_file, pdtb_file, output_file, stats)
        
        else:
            if conll_name.endswith(".v2_gold_coref"):
                pdtb_file = os.path.join(pdtb_directory, conll_name[: -14])
                output_file = os.path.join(output_directory, conll_name[: -14] + ".json")
                process_file(conll_parser, conll_path, pdtb_file, output_file, stats)
    
    
def process_file(conll_parser, conll_file, pdtb_file, output_file, stats):
    """
    Process a file of CoNLL-2011 data.
    
    Args:
        conll_parser: An ConllParser object.
        conll_file: The path of a CoNLL-2011 file.
        pdtb_file: The path of the equivalent PDTB v2 file.
        output_file: The path of the output file.
        stats: A dict using to record statistics.
    """
    
    try:
        coreference_groups = conll_parser.process_conll_file(conll_file, pdtb_file)  
    
    except ValueError as err:
        logger.error(err)
        return
    
    stats["files"] = stats["files"] + 1
    stats["groups"] = stats["groups"] + len(coreference_groups)     
    
    for coreference_group in coreference_groups:
        stats["coreferences"] = stats["coreferences"] + len(coreference_group)
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(coreference_groups, ensure_ascii=False, indent=4, sort_keys=False))
        outfile.flush()
        outfile.close()


class ConllParser(object):
    """
    Class which parses CoNLL-2011 files and extracts coreference data.
    """
    
    def process_conll_file(self, conll_file, pdtb_file):
        """
        Process a CoNLL-2011 Shared Task datafile.
        
        Note that CoNLL can have extra \n compared to PDTB.
        and that PDTB can have two \n where CoNLL has one.
        
        Coreference are represented by a dict with start, end and text fields.
        The start and end fields are character offsets in the PDTB file.
        
        Args:
            conll_file: The path of a CoNLL-2011 file.
            pdtb_file: The path of the equivalent PDTB v2 file.
        
        Returns:
            A list of coreference groups. Each coreference group is a list of spans.
        """
        
        self.filename = conll_file
        
        coreferences = {}       # Map between coreference_ids and coreference spans.
        self.pdtb_index = 0     # character offset into PDTB file 
        self.conll_index = 0    # character offset into CONLL file
        
        self.pdtb_text = self.read_pdtb_file(pdtb_file)
        tree = ET.parse(conll_file)
        
        for element in tree.getroot():
            if element.tag == "TEXT":
                self.process_element(element, coreferences)
        
        coreference_groups = []
        
        for coreference_group in coreferences.values():
            coreference_groups.append(coreference_group)
        
        return coreference_groups
    
    
    def process_element(self, element, coreferences):
        """
        Process an XML element and accumulate coreferences.
        
        Args:
            element: An xml.etree.cElementTree.Element object.
            coreferences: A dict mapping coreference IDs to coreference spans.
        """
        
        if element.text is not None:
            element_text = self.process_text(element.text)
            
            while len(self.pdtb_text[self.pdtb_index].strip()) == 0:
                self.pdtb_index += 1
            
            start = self.pdtb_index
            self.soft_match(element_text)
            end = self.pdtb_index
            
            if element.tag == "COREF":              
                if element.attrib["TYPE"] == "IDENT":
                    if start is not None and end is not None:
                        coreference_id = element.attrib["ID"]
                        coreference = {}
                        coreference["start"] = start + PDTB_PREFIX_LENGTH
                        coreference["end"] = end + PDTB_PREFIX_LENGTH
                        coreference["text"] = self.pdtb_text[start : end]
                        
                        if coreference_id in coreferences:
                            coreferences[coreference_id].append(coreference)
                        else:
                            coreferences[coreference_id] = [coreference]
        
        for child in element:
            self.process_element(child, coreferences)
        
        if element.tail is not None:
            text = self.process_text(element.tail)
            self.soft_match(text)
    
    
    def soft_match(self, conll_element_text):
        """
        Advance self.pdtb_index until a match is achieved for the supplied text.
        
        Args:
            conll_element_text: A text string.
        """
        
        for conll_line in conll_element_text.splitlines():
            if self.pdtb_index < len(self.pdtb_text):
                self.soft_match_line(conll_line)
    
    
    def soft_match_line(self, conll_line):
        """
        Advance self.pdtb_index until a match is achieved for the supplied text.
        
        Args:
            conll_line: A text string.
            
        Raises:
            ValueError if a match cannot be found.
        """
        
        conll_line = conll_line.strip()
        
        while len(self.pdtb_text[self.pdtb_index].strip()) == 0:
            self.pdtb_index += 1
            
            if self.pdtb_index >= len(self.pdtb_text):
                return
        
        if conll_line == "." and self.pdtb_text[self.pdtb_index] != ".":
            return
                 
        pdtb_line = self.pdtb_text[self.pdtb_index: self.pdtb_text.find("\n", self.pdtb_index)].strip()
        
        if self.pdtb_text.startswith(conll_line, self.pdtb_index):
            self.pdtb_index += len(conll_line)
            return
        
        elif conll_line.startswith(pdtb_line):
            self.pdtb_index += len(pdtb_line)
            self.soft_match(conll_line[len(pdtb_line) :])
        
        else:
            raise ValueError("ERROR: No match in: " + self.filename + "\nfor CoNLL line: " + conll_line + "\nand PDTB line:  " + pdtb_line)
    
    
    def process_text(self, text):
        """
        Process text from CoNLL-2011 to ensure a match with PDTB v2.0. If the soft_match_line 
        method fails to achieve a match it raises a ValueError. These issues  can usually be 
        resolved by ad-hoc adjustments here or editing CoNLL-2011 files.
        
        Args:
            text: A text string.
        
        Returns:
           A text string.
        """
        
        text = text.replace(" n't", "n't")
        text = text.replace(" %", "%")
        text = text.replace("-AMP-", "&")
        text = text.replace("-RCB-", "}")
        text = text.replace("-LCB-", "{")
        text = text.replace("-RRB-", ")")
        text = text.replace("-LRB-", "(")
        text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
        text = text.replace("\\*", " *")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")
        text = text.replace("`` ", "\"")
        text = text.replace("``", "\"")
        text = text.replace(" ''", "\"")
        text = text.replace("''", "\"")
        text = text.replace(" ' ", "' ")
        text = text.replace(" 'd", "'d")
        text = text.replace(" 'l", "'l")
        text = text.replace(" 'r", "'r")
        text = text.replace(" 's", "'s")
        text = text.replace(" 'v", "'v")
        text = text.replace(" 'm", "'m")
        text = text.replace("s '", "s'")
        text = text.replace("{ ", "{")
        text = text.replace(" }", "}")
        text = text.replace("( ", "(")
        text = text.replace(" )", ")")
        text = text.replace("$ ", "$")
        text = text.replace(" ;", ";")
        text = text.replace("can not", "cannot")
        text = text.replace(" :", ":")
        text = text.replace("....", ". . . .")
        text = text.replace("...", " . . .")
        text = text.replace("& Co.'s", "& Co. 's")
        text = text.replace("Corp.'s", "Corp. 's")
        text = text.replace("com-pany", "com- pany")
        text = text.replace(" - ", "-")
        text = text.replace(" / ", "/")
        text = text.replace("` ", "`")
        text = text.replace(" 'S", "'S")
        text = text.replace(" # ", " #")
        text = text.replace("?'", "? '")
        
        text = text.replace("pro-and anti-abortionists", "pro- and anti-abortionists")
        text = text.replace("Co.'s", "Co. 's")
        text = text.replace("earnings,'", "earnings, '")
        text = text.replace("says, `", "says, '")
        text = text.replace(",' everybody", ", ' everybody")
        text = text.replace("t\" and brazen", "t \" and brazen")
        text = text.replace("bad,' so", "bad, ' so")
        text = text.replace("Inc.'s", "Inc. 's")
        text = text.replace("level of focus", "level offocus")
        text = text.replace("and say, `", "and say, '")
        text = text.replace("up the figures -", "up the figures-")
        text = text.replace("niche-itis,\" ", "niche-itis,\"")
        
        if text == "-and ":
            text = "- and "
        
        text = text.replace(",'", ", '")
        text = text.replace(". '\nNow", ".'\nNow")         
         
        text = text.replace("not seek . . . to", "not seek. . .to")
        text = text.replace("substantial long positions that is", "substantial long positionsthat is")
        text = text.replace("have started con-structively", "have started con- structively")
        text = text.replace("Mortgage -, Asset", "Mortgage-, Asset")
        text = text.replace("and say, 'Let", "and say, `Let")
        text = text.replace("Byrne of the U.S..", "Byrne of the U.S.")
        text = text.replace("is not insured. . . .", "is not insured . . . .")
        text = text.replace("nothin '.\"", "nothin'.\"")
        text = text.replace("possible moment when other", "possible momentwhen other")
        text = text.replace("30 billion *", "30 billion*")
        text = text.replace("44 cents  * *", "44 cents**")
        text = text.replace("* Includes", "*Includes")
        text = text.replace("* * Year", "**Year")
        text = text.replace("JURY'S", "JURY`S")
        text = text.replace("market: 8.60%", "market:8.60%")
        text = text.replace("turban 10 years", "turban10 years")
        text = text.replace("translator for a Russian", "translatorfor a Russian")
        text = text.replace("not sure\") and Bridget", "not sure \") and Bridget")
        text = text.replace("`Liz. ' . . .", "`Liz. '. . .")

        text = text.replace("it s ", "its ")
        text = text.replace("tune . . . '", "tune . . .'")
        text = text.replace(": \" . . . ", ": \". . . ")
        text = text.replace("gon na ", "gonna ")
        text = text.replace("FFr 27.68", "FFr27.68")
        text = text.replace("* Not counting", "*Not counting")
        text = text.replace("* * With ", "**With ")
        text = text.replace("Gaming company", "gaming company")
        text = text.replace("THE U.S..", "THE U.S.")
        text = text.replace("company-and", "company- and")
        text = text.replace("Victorian in its influence. . . .", "Victorian in its influence....")
        text = text.replace("* For ", "*For ")
        text = text.replace("IS N'T", "ISN'T")
        text = text.replace("'T- is the", "'Tis the")
        text = text.replace("I -'m coming-down-your-throat", "I'm-coming-down-your-throat")
        text = text.replace("Cos.'", "Cos. '")
        text = text.replace("receiving US$500,000", "receiving S$500,000")
        text = text.replace("bail of US$1 million", "bail of S$1 million")
        text = text.replace("U.S..", "U.S.")
        text = text.replace(".} the previous", "} the previous")
        text = text.replace(" medium-and long", " medium- and long")
        text = text.replace("Americannotions", "American notions")
        text = text.replace("ended . . . sometime", "ended. . . sometime")
        text = text.replace("17-to 18-year", "17- to 18-year")
        text = text.replace("ask `What else", "ask 'What else")
        text = text.replace("Australia Ltd.'s", "Australia Ltd. 's")
        
        
            
        text = text.replace("Net loss: $1.30 billion  *", "Net loss: $1.30 billion*")
        text = text.replace("or a service, The Movie Channel", "or a sister service, The Movie Channel")
        text = text.replace("price indexes (1982 = 100)", "price indexes (1982=100)")
        text = text.replace("UniHealth America Inc. (", "UniHealth America Inc.(")
        
        if len(text) == 8:
            text = text.replace("Bard/EMS", "Bard/ EMS")
        
        if self.filename.endswith("wsj_2351.v2_gold_coref"):
            text = text.replace(", '\"", ",'\"")
        
        return text
    
    
    @staticmethod
    def read_pdtb_file(filename):
        """
        Read a PDTB v2 file and remove the 9 character prefix.
        
        Args:
            filename: The path to PDTB v2 file.
        
        Returns:
           A text string.
        """
        
        with open(filename, encoding="utf-8") as infile:
            text = infile.read()
            return text[PDTB_PREFIX_LENGTH :]


if __name__ == "__main__":
    main()
