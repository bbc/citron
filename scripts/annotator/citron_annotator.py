# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This a GUI application which adds annotations to text documents to identify
the source, cue and content of quotes and coreferences for the sources.
The data is stored in a Citron format JSON data file.
"""

import tkinter as tk
from tkinter.font import Font 
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Frame
from tkinter.ttk import Button
import json
import sys
import os

from tkreadonly import ReadOnlyText

from citron.cue import CueClassifier
from citron import utils

SELECTION_COLOUR            = "magenta"
SELECTED_CUE_COLOUR         = "yellow"
SELECTED_SOURCE_COLOUR      = "green"
SELECTED_COREFERENCE_COLOUR = "orange"
SELECTED_CONTENT_COLOUR     = "cyan"
DESELECTED_CUE_COLOUR       = "pink"
DESELECTED_COLOUR           = "#DFCDFCDFC"


class Annotator(Frame):
    """
    Quotes consist of one or more sources, a cue and one or more content spans.
    They may also have coreferences for the sources.
   
    Internally, the spans are recorded as tags in the text widget with names of the 
    form:
    
        <span_name>_i-j
        
    where i is the quote index and j is the span index
    
        e.g. source_0-1
    
    In the output file the spans are specified using start and end character offsets.
    """
    
    def __init__(self, input_directory, output_directory, model_path=None):
        Frame.__init__(self)
        self.grid()
        self.create_widgets()
        self.selected_quote_index = None
        self.out_file_path = None
        self.model_path = model_path
        self.is_unsaved = False
        self.input_directory = input_directory
        
        if os.path.isdir(output_directory):
            self.output_directory = output_directory
        else:
            print("Error: Unable to find output directory:", output_directory)
            sys.exit(0)
    
    
    def create_widgets(self):
        font = Font(size=14)
        self.text_box = ReadOnlyText(self, height=20, width=80, wrap="word", font = font)
        self.text_box.grid(row=0, column=0, columnspan=5, stick=tk.NSEW)
        self.text_box.tag_config(tk.SEL, background = SELECTION_COLOUR)
        
        self.cue_button = Button(self, text="Select or Add Cue", command=self.select_or_add_cue, state=tk.DISABLED)
        self.cue_button.grid(row=1, column=0, sticky=tk.W)
        
        self.source_button = Button(self, text="Add Source", command=self.add_source, state=tk.DISABLED)
        self.source_button.grid(row=1, column=1, sticky=tk.W)
        
        self.coreference_button = Button(self, text="Add Coreference", command=self.add_coreference, state=tk.DISABLED)
        self.coreference_button.grid(row=1, column=2, sticky=tk.W)
         
        self.content_button = Button(self, text="Add Content", command=self.add_content, state=tk.DISABLED)
        self.content_button.grid(row=1, column=3, sticky=tk.W)
        
        self.amend_button = Button(self, text="Amend Span", command=self.amend_span, state=tk.DISABLED)
        self.amend_button.grid(row=1, column=4, sticky=tk.W)
        
        self.open_button = Button(self, text="Open File", command=self.check_before_open)
        self.open_button.grid(row=2, column=0, sticky=tk.W)
        
        self.save_button = Button(self, text="Save File", command=self.save_file, state=tk.DISABLED)
        self.save_button.grid(row=2, column=1, sticky=tk.W)      
         
        self.quit_button = Button(self, text="Quit", command=self.check_before_quit)
        self.quit_button.grid(row=2, column=2, sticky=tk.W)
        
        self.remove_button = Button(self, text="Remove Span", command=self.remove_span, state=tk.DISABLED)
        self.remove_button.grid(row=2, column=4, sticky=tk.W)  
    
    
    def select_or_add_cue(self):
        first, last = self.get_selection_indices()
        
        if first == last:
            self.clear_selection()
            return
        
        overlapping_tags = self.get_overlapping_tags(first, last)
        
        if len(overlapping_tags) == 0:
            quote_index = None
        
        else:
            for overlapping_tag in overlapping_tags:
                if overlapping_tag.startswith("cue"):
                    quote_index = self.get_quote_index(overlapping_tag)
                    break
            else:
                messagebox.showinfo("Select Cue", "Cues cannot overlap existing spans", icon=messagebox.ERROR)
                self.clear_selection()
                return
        
        #================================================================#
        # quote_index      | selected_quote_index     | action           #
        # -------------------------------------------------------------- #
        # None             | None                     | create new tag   #
        # None             | current                  | create new tag   #
        # current          | current                  | ignore selection #
        # new              | current                  | change selection #
        #================================================================#
        
        if quote_index is None:
            self.deselect_quote(self.selected_quote_index)
            quote_index = self.get_next_quote_index()
            self.add_tag(first, last, quote_index, "cue", True)
        else:
            if quote_index == self.selected_quote_index:
                self.clear_selection()
                return
            else:
                self.deselect_quote(self.selected_quote_index)
                self.select_quote(quote_index)

        self.selected_quote_index = quote_index
        self.configure_edit_buttons(tk.NORMAL)
        self.clear_selection()   
    
    
    def add_source(self):
        self.add_selection_as_tag("source")   
    
    
    def add_coreference(self):
        self.add_selection_as_tag("coreference")
    
    
    def add_content(self):
        self.add_selection_as_tag("content")
    
    
    def add_selection_as_tag(self, prefix):
        if self.selected_quote_index is None:
            span_type = prefix[0].upper() + prefix[1:]
            messagebox.showinfo("Add " + span_type, "A cue must be selected before adding a " + prefix, icon=messagebox.ERROR)
            return
        
        if prefix != "content":
            existing_tags = self.get_tags(self.selected_quote_index, prefix)
            
            if len(existing_tags) > 0:
                span_type = prefix[0].upper() + prefix[1:]
                messagebox.showerror("A " + span_type + " has already been specified")    
                return

        first, last = self.get_selection_indices()
        
        
        if self.is_valid_tag(first, last, prefix, True):
            self.add_tag(first, last, self.selected_quote_index, prefix, True)
            
        self.clear_selection()
    
    
    def is_valid_tag(self, first, last, prefix, invoke_dialog=False, tag_to_ignore=None):        
        if first == last:
            return False
         
        overlapping_tags = self.get_overlapping_tags(first, last)
        
        for overlapping_tag in overlapping_tags:
            if tag_to_ignore is None or tag_to_ignore != overlapping_tag:
                overlapping_tag_quote_index = self.get_quote_index(overlapping_tag)
                
                if overlapping_tag_quote_index == self.selected_quote_index:
                    if invoke_dialog:
                        span_type = prefix[0].upper() + prefix[1:]
                        messagebox.showerror("Add " + span_type, "Quote spans cannot overlap")
                    
                    return False
                
                overlapping_tag_prefix = self.get_prefix(overlapping_tag)            
                
                if prefix == "coreference":
                    if overlapping_tag_prefix == "cue":
                        if invoke_dialog:
                            span_type = prefix[0].upper() + prefix[1:]
                            messagebox.showerror("Add " + span_type, "Coreferences cannot overlap with cues")
                            
                        return False
                    else:
                        return True
                    
                elif prefix == "source":
                    if overlapping_tag_prefix in ("cue", "content"):
                        if invoke_dialog:
                            span_type = prefix[0].upper() + prefix[1:]
                            messagebox.showerror("Add " + span_type, "Sources cannot overlap with cues or content")
                        
                        return False
                    else:
                        return True
                
                else:
                    if invoke_dialog:
                        span_type = prefix[0].upper() + prefix[1:]
                        messagebox.showerror("Add " + span_type, prefix + "s cannot overlap with " + overlapping_tag_prefix + "s")
                    
                    return False
            
        return True
    
    
    def get_tags(self, quote_index=None, prefix=None):
        tags = []
        
        for tag in self.text_box.tag_names():
            if tag == tk.SEL:
                continue
            
            if prefix is None or tag.startswith(prefix):
                if quote_index is None or quote_index == self.get_quote_index(tag):
                    tags.append(tag)
        
        return tags
    
    
    def get_character_indices(self, tag):
        start = self.get_character_index(tag + ".first")
        end =   self.get_character_index(tag + ".last")
        return start, end
    
    
    def get_text(self, tag):
        return self.text_box.get(tag + ".first", tag + ".last")
    
    
    def get_character_index(self, coord):
        position = self.text_box.count("1.0", coord)
        
        if position is None:
            return 0
        else:
            return position[0]
    
    
    def get_index(self, character_index):
        return "1.0+" + str(character_index) + "c"
    
    
    def get_selection_indices(self):
        # first and last are strings e.g. "n.m" where n is line number and m is column number
        first = self.text_box.index("sel.first")
        last  = self.text_box.index("sel.last")
        return first, last
    
    
    def add_tag(self, first, last, quote_index, prefix, is_selected):
        colour = self.get_selected_colour(prefix, is_selected)
        span_index = self.get_next_span_index(prefix, quote_index)
        tag = self.get_tag(prefix, quote_index, span_index)
        self.text_box.tag_add(tag, first, last)
        self.text_box.tag_config(tag, background=colour)        
        self.text_box.tag_raise(tag)
        self.is_unsaved = True
    
    
    def get_selected_colour(self, prefix, is_selected):
        if is_selected:
            if prefix == "cue":
                return SELECTED_CUE_COLOUR
            
            elif prefix == "source":
                return SELECTED_SOURCE_COLOUR
            
            elif prefix == "coreference":
                return SELECTED_COREFERENCE_COLOUR
            
            elif prefix == "content":
                return SELECTED_CONTENT_COLOUR
            
            else:
                return None
        
        else:
            if prefix == "cue":
                return DESELECTED_CUE_COLOUR
            
            else:
                return DESELECTED_COLOUR
    
    
    def amend_span(self):
        first, last = self.get_selection_indices()
        
        if first == last:
            self.clear_selection()
            return
        
        overlapping_tags = self.get_overlapping_tags(first, last, self.selected_quote_index)
        
        if len(overlapping_tags) == 0:
            self.clear_selection()
            return
        
        elif len(overlapping_tags) > 1:
            messagebox.showerror("Amend Span", "More than one tag has been selected")
            self.clear_selection()
            return
        
        tag = overlapping_tags[0]
        prefix = self.get_prefix(tag)
        
        if self.is_valid_tag(first, last, prefix, True, tag):
            self.text_box.tag_delete(tag)
            self.add_tag(first, last, self.selected_quote_index, prefix, True)
            
        self.clear_selection()
        self.is_unsaved = True
    
    
    def remove_span(self):
        selection_first, selection_last = self.get_selection_indices()
        
        if selection_first == selection_last:
            self.clear_selection()
            return
        
        overlapping_tags = self.get_overlapping_tags(selection_first, selection_last, self.selected_quote_index)
        
        if len(overlapping_tags) == 0:
            self.clear_selection()
            return
        
        elif len(overlapping_tags) > 1:
            messagebox.showerror("Remove Span", "More than one tag has been selected")
            self.clear_selection()
            return
        
        tag = overlapping_tags[0]
        
        if tag.startswith("cue"):
            if self.cue_has_additional_spans(self.selected_quote_index):            
                if messagebox.askokcancel("Remove Cue", "Warning: This will remove all spans of the quote", icon=messagebox.WARNING):
                    for tag in self.get_tags(self.selected_quote_index):            
                        self.text_box.tag_delete(tag)
            else:
                self.text_box.tag_delete(tag)
        else:    
            self.text_box.tag_delete(tag)
        
        self.clear_selection()
        self.is_unsaved = True
    
    
    def get_overlapping_tags(self, first, last, quote_index=None):
        sel_start = self.get_character_index(first)
        sel_end   = self.get_character_index(last)
        overlapping_tags = []
        
        for tag in self.text_box.tag_names():
            if tag == tk.SEL:
                continue
        
            if quote_index is None or quote_index == self.get_quote_index(tag):
                tag_first = self.get_first(tag)
                tag_last = self.get_last(tag)
                tag_start = self.get_character_index(tag_first)
                tag_end   = self.get_character_index(tag_last) 
                
                if tag_start < sel_end and  tag_end > sel_start:
                    overlapping_tags.append(tag)
        
        return overlapping_tags
    
    
    def select_quote(self, quote_index):
        for tag in self.text_box.tag_names():
            if tag == tk.SEL:
                continue
            
            if self.get_quote_index(tag) == quote_index:
                if tag.startswith("cue"):
                    self.text_box.tag_configure(tag, background = SELECTED_CUE_COLOUR)
                
                if tag.startswith("source"):
                    self.text_box.tag_configure(tag, background = SELECTED_SOURCE_COLOUR)
                
                if tag.startswith("coreference"):
                    self.text_box.tag_configure(tag, background = SELECTED_COREFERENCE_COLOUR)
                
                if tag.startswith("content"):
                    self.text_box.tag_configure(tag, background = SELECTED_CONTENT_COLOUR)
                    
                self.text_box.tag_raise(tag)
    
      
    def deselect_quote(self, quote_index):
        for tag in self.text_box.tag_names():
            if tag == tk.SEL:
                continue

            if self.get_quote_index(tag) == quote_index:
                if tag.startswith("cue"):
                    self.text_box.tag_configure(tag, background = DESELECTED_CUE_COLOUR)
                
                if tag.startswith("source"):
                    self.text_box.tag_configure(tag, background = DESELECTED_COLOUR)
                
                if tag.startswith("coreference"):
                    self.text_box.tag_configure(tag, background = DESELECTED_COLOUR)
                
                if tag.startswith("content"):
                    self.text_box.tag_configure(tag, background = DESELECTED_COLOUR)
    
    
    def get_tag(self, prefix, quote_index, span_index):
        return prefix + "_" + str(quote_index) + "-" + str(span_index)
    
    
    def get_prefix(self, tag):
        end = tag.find("_")
        return tag[0 : end]
    
    
    def get_quote_index(self, tag):
        start = tag.find("_") + 1
        end = tag.find("-")
        
        if end == -1:
            end = len(tag)
        
        return int(tag[start : end])
    
    
    def get_span_index(self, tag):
        start = tag.find("-") + 1
        return int(tag[start:])
    
    
    def get_first(self, tag):
        return self.text_box.index(tag +".first")
    
    
    def get_last(self, tag):
        return self.text_box.index(tag +".last")
    
    
    def get_next_quote_index(self):
        next_index = 0
        
        for tag in self.text_box.tag_names():
            if tag == tk.SEL:
                continue
            
            index = self.get_quote_index(tag)
            next_index = max(index + 1, next_index)
                  
        return next_index
    
    
    def get_next_span_index(self, prefix, quote_index):
        span_count = 0
        
        for tag in self.text_box.tag_names():
            if tag.startswith(prefix + "_" + str(quote_index)):
                span_count += 1
                     
        return span_count
    
    
    def cue_has_additional_spans(self, quote_index):
        for tag in self.get_tags(quote_index):
            if not tag.startswith("cue"):
                return True
        
        return False
    
    
    def check_before_open(self):
        if self.is_unsaved:
            if messagebox.askokcancel("Open File", "There are unsaved annotations - are you sure you want to open a new file?", icon=tk.messagebox.WARNING):
                self.open_file() 
        else:
            self.open_file() 
    
    
    def open_file(self):        
        in_file_path = filedialog.askopenfilename(
            initialdir = self.input_directory,
            title = "Select text file"
        )
        
        # Remove dialog immediately
        self.update()
        self.clear_all()
        in_file_name = os.path.split(in_file_path)[1]
        index = in_file_name.rfind(".")
                
        if index == -1:
            out_file_name = in_file_name + ".json"
        else:
            out_file_name = in_file_name[0 : index] + ".json"
        
        app.master.title(out_file_name)
        self.out_file_path = os.path.join(self.output_directory, out_file_name)
        
        if os.path.isfile(self.out_file_path):
            print("Loading existing annotations file:", self.out_file_path)
            self.load_annotations_file(self.out_file_path)
        
        else:
            print("Loading input file:", in_file_path)
            
            with open(in_file_path, encoding="utf-8") as infile:
                self.text = infile.read().strip()
                self.text_box.insert("1.0", self.text)
                
                if self.model_path is not None:
                    predict_cues = messagebox.askquestion("Open File", "Predict quote-cues for input text?", icon=messagebox.QUESTION)
                    self.update()
        
                    if predict_cues == "yes":
                        self.predict_cues(self.text)
        
        self.save_button.config(state=tk.NORMAL)
        self.cue_button.config(state=tk.NORMAL)
        self.clear_selection()
        self.is_unsaved = False
    
    
    def predict_cues(self, text):
        if not hasattr(self, "cue_classifier"):
            self.cue_classifier = CueClassifier(self.model_path)
        
        if not hasattr(self, "nlp"):
            print("Loading spacy model")
            self.nlp = utils.get_parser()
        
        doc = self.nlp(text)
        cues = self.cue_classifier.predict_cues_and_labels(doc)[0]        
        quote_index = 0
        
        for cue in cues:
            first = self.get_index(cue.start_char)
            last  = self.get_index(cue.end_char)
            
            if self.is_valid_tag(first, last, "cue"):
                if quote_index == 0:
                    self.add_tag(first, last, quote_index, "cue", False)
                
                else:
                    self.add_tag(first, last, quote_index, "cue", False)
            
            else:
                print("Error: Invalid cue", first, last)
                continue
            
            quote_index += 1
    
    
    def load_annotations_file(self, file_path):
        """
        Load existing annotations
        
        """
        with open(file_path, encoding="utf-8") as infile:
            data = json.load(infile)
            self.text = data["text"]
            self.text_box.insert("1.0", self.text)
            quotes = data["quotes"]
            quote_index = 0
            
            for quote in quotes:
                if not "cue" in quote:
                    print("Invalid entry:", json.dumps(quote))
                    continue
                
                cue = quote["cue"]
                first = self.get_index(cue["start"])
                last  = self.get_index(cue["end"])
                
                if self.is_valid_tag(first, last, "cue"):           
                    self.add_tag(first, last, quote_index, "cue", False)
                
                else:
                    print("Error: Invalid cue", first, last)
                    continue
                
                if "source" in quote:
                    source = quote["source"]
                    first = self.get_index(source["start"])
                    last  = self.get_index(source["end"])
                    
                    if self.is_valid_tag(first, last, "source"):           
                        self.add_tag(first, last, quote_index, "source", False)
                    
                    else:
                        print("Error: Invalid source", first, last)
                        continue
                
                if "coreference" in quote:
                    coreference = quote["coreference"]
                    first = self.get_index(coreference["start"])
                    last  = self.get_index(coreference["end"])
                    
                    if self.is_valid_tag(first, last, "coreference"):           
                        self.add_tag(first, last, quote_index, "coreference", False)
                    
                    else:
                        print("Error: Invalid coreference", first, last)
                        continue
                
                for content in quote["contents"]:
                    first = self.get_index(content["start"])
                    last  = self.get_index(content["end"])
                    
                    if self.is_valid_tag(first, last, "content"):           
                        self.add_tag(first, last, quote_index, "content", False)
                    
                    else:
                        print("Error: Invalid content", first, last)
                        continue

                quote_index += 1
                
        print("Info: Loaded", quote_index, "existing annotations")
    
    
    def get_quotes(self):
        quotes = []
        
        for tag_name in self.text_box.tag_names():
            if tag_name == tk.SEL:
                continue
            
            if tag_name.startswith("cue"):
                quote_index = self.get_quote_index(tag_name)
                start, end = self.get_character_indices(tag_name)
                text = self.get_text(tag_name)
                
                cue = {}
                cue["start"] = start 
                cue["end"] = end
                cue["text"] = text
                sources = []
                coreferences = []
                contents = []
                
                for tag in self.get_tags(quote_index, "source"):
                    start, end = self.get_character_indices(tag)
                    text = self.get_text(tag)
                    source = {}
                    source["start"] = start 
                    source["end"] = end
                    source["text"] = text
                    sources.append(source)
                
                for tag in self.get_tags(quote_index, "coreference"):
                    start, end = self.get_character_indices(tag)
                    text = self.get_text(tag)
                    coreference = {}
                    coreference["start"] = start 
                    coreference["end"] = end
                    coreference["text"] = text
                    coreferences.append(coreference)
                       
                for tag in self.get_tags(quote_index, "content"):
                    start, end = self.get_character_indices(tag)
                    text = self.get_text(tag)
                    content = {}
                    content["start"] = start 
                    content["end"] = end
                    content["text"] = text
                    contents.append(content)
                
                quote = {}
                quote["cue"] = cue
                quote["sources"]  = sorted(sources, key=lambda dct: dct["start"])
                quote["contents"] = sorted(contents, key=lambda dct: dct["start"])
                
                if len(coreferences) > 0:
                    quote["coreferences"] = sorted(coreferences, key=lambda dct: dct["start"])
                
                quotes.append(quote)
        
        return sorted(quotes, key=lambda dct: dct["cue"]["start"])
    
    
    def save_file(self):
        data = {}
        data["quotes"] = self.get_quotes()
        data["text"] = self.text
           
        if os.path.isfile(self.out_file_path):
            backup_file_path = self.out_file_path.replace(".json", ".bak")
            os.rename(self.out_file_path, backup_file_path)
           
        with open(self.out_file_path, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False))
            outfile.flush()
            outfile.close()
            self.is_unsaved = False
    
    
    def clear_selection(self):
        self.text_box.tag_remove(tk.SEL, "1.0", tk.END)
        self.text_box.tag_raise(tk.SEL)
    
     
    def clear_all(self):
        self.text_box.delete("1.0", tk.END)
        
        # Removes tags and highlights
        for tag in self.text_box.tag_names():
            self.text_box.tag_delete(tag)
    
    
    def configure_edit_buttons(self, state):
        self.source_button.config(state=state)
        self.coreference_button.config(state=state)
        self.content_button.config(state=state)
        self.amend_button.config(state=state)
        self.remove_button.config(state=state)
    
    
    def check_before_quit(self):
        if self.is_unsaved:
            if messagebox.askokcancel("Quit Annotator", "There are unsaved annotations - are you sure you want to quit?", icon=tk.messagebox.WARNING):
                self.quit()
        else:
            self.quit()


if len(sys.argv) == 3:
    app = Annotator(sys.argv[1], sys.argv[2])

elif len(sys.argv) == 4:
    app = Annotator(sys.argv[1], sys.argv[2], sys.argv[3])

else:
    print("Usage:  annotator <Input directory> <Output directory> [<Citron model directory>]")
    sys.exit(0)    

app.master.title("Citron Annotator")
app.mainloop()
