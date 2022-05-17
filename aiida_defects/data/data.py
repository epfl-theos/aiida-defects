from string import Template

import numpy as np
import pandas as pd
from aiida.common.exceptions import ValidationError
from aiida.common.utils import prettify_labels
from aiida.orm import ArrayData
from aiida_defects.formation_energy.chemical_potential.utils import Order_point_clockwise

class StabilityData(ArrayData):
	'''
	Class to represent the stability region of a compound and all its attributes (vertice of stability region, etc.)
	The visualization only works for 2D stability region, i.e. for ternary compounds. For compounds with more than 3 elements,
	some chemical potentials have to be set at certain values so that a 'slice' of the stability region can be plotted.
	'''
	def set_data(self, matrix_of_constraints, indices, columns, stability_vertices, compound, dependent_element, property_map=None):
		self.set_array('matrix', matrix_of_constraints)
		if stability_vertices.shape[1] == 3:
			self.set_array('vertices', Order_point_clockwise(stability_vertices)) # vertices include the coordinate of the dependent element as well
		else:
			raise ValueError('The stability vertices must be an Nx3 array where N is the number of compounds in the phase diagram.'
							'The 3 columns corresponds to the chemical potentials of each each elements'
							)
		self.column = columns
		self.compound = compound
		self.dependent_element = dependent_element
		self.index = indices
		self.property_map = property_map

	def get_constraints(self):
		return self.get_array('matrix')

	def get_vertices(self):
		return self.get_array('vertices')

	@property
	def index(self):
		return self.get_attribute('index')

	@index.setter 
	def index(self, value):
		self._set_index(value)

	def _set_index(self, value):
		'''
		The equation for the dependent element is the same as that of the compound, therefore we can replace the 
		dependent element in indices by the compound name which is useful later for plotting purpose.
		'dependent_element' and 'compound' have to be set first
		'''
		idx = [l if l != self.dependent_element else self.compound for l in value]
		self.set_attribute('index', idx)

	@property
	def column(self):
		return self.get_attribute('column')

	@column.setter
	def column(self, value):
		self.set_attribute('column', value)

	@property
	def compound(self):
		return self.get_attribute('compound')

	@compound.setter
	def compound(self, value):
		self.set_attribute('compound', value)

	@property
	def dependent_element(self):
		return self.get_attribute('dependent_element')

	@dependent_element.setter
	def dependent_element(self, value):
		self.set_attribute('dependent_element', value)

	@property
	def property_map(self):
		return self.get_attribute('property_map')

	@property_map.setter
	def property_map(self, value):
		self.set_attribute('property_map', value)

	def _get_stabilityplot_data(self):
		'''
		Get data to plot a stability region 
		Make sure the data are suitable for 2D plot. TO BE DONE.
		'''

		x_axis = np.arange(-10, 0.01, 0.05)
		x = []
		y = []

		### Lines corresponding to each constraint associated with each compound
		M = self.get_constraints()
		# print(pd.DataFrame(M, index=self.index, columns=self.column))

		### The matrix M has the shape Nx3 where N is the number of compounds and elemental phases. 
		### The 1st column is the 'x axis', 2nd column 'y axis' and 3rd column is related to the formation energy
		for l in M:
			if l[1] == 0.0: # vertical line
				x.append([l[2]/l[0]]*len(x_axis))
				y.append(x_axis)
			else:
				x.append(x_axis)
				y.append((l[2]-l[0]*x_axis)/l[1])

		x = np.array(x)
		y = np.array(y)
		vertices = self.get_vertices()

		plot_info = {}
		plot_info['x'] = x
		plot_info['y'] = y
		plot_info['vertices'] = vertices
		boundary_lines = set()

		### Find the boundary compounds, i.e. compounds whose corresponding lines form the edge of the stability region
		for vtx in plot_info['vertices']:
			# Check if a vertex is on the line, i.e its cooridinates verify the equation corresponding to that line
			mask = np.abs(M[:,:2]@np.reshape(vtx[:2], (2,1))[:,0] - M[:,-1]) < 1E-4
			# Check all lines that pass through the vertex vtx
			idx = [i for i, _ in enumerate(mask) if _]
			# Find the corresponding name of the compound associate with that lines
			boundary_lines = boundary_lines.union(set([self.index[j] for j in idx]))
		plot_info['boundary_lines'] = boundary_lines

		if self.property_map:
			plot_info['grid'] = self.property_map['points_in_stable_region']
			plot_info['property'] = self.property_map['property']

		return plot_info

	def _matplotlib_get_dict(
		self,
		main_file_name='',
		comments=True,
		title='',
		legend_location=None,
		x_max_lim=None,
		x_min_lim=None,
		y_max_lim=None,
		y_min_lim=None,
		prettify_format=None,
		**kwargs
		):  # pylint: disable=unused-argument
		"""
		Prepare the data to send to the python-matplotlib plotting script.

		:param comments: if True, print comments (if it makes sense for the given
		    format)
		:param plot_info: a dictionary
		:param setnumber_offset: an offset to be applied to all set numbers
		    (i.e. s0 is replaced by s[offset], s1 by s[offset+1], etc.)
		:param color_number: the color number for lines, symbols, error bars
		    and filling (should be less than the parameter MAX_NUM_AGR_COLORS
		    defined below)
		:param title: the title
		:param legend_location: the position of legend
		:param y_max_lim: the maximum on the y axis (if None, put the
		    maximum of the bands)
		:param y_min_lim: the minimum on the y axis (if None, put the
		    minimum of the bands)
		:param y_origin: the new origin of the y axis -> all bands are replaced
		    by bands-y_origin
		:param prettify_format: if None, use the default prettify format. Otherwise
		    specify a string with the prettifier to use.
		:param kwargs: additional customization variables; only a subset is
		    accepted, see internal variable 'valid_additional_keywords
		"""
		# pylint: disable=too-many-arguments,too-many-locals

		# Only these keywords are accepted in kwargs, and then set into the json
		valid_additional_keywords = [
		    'bands_color',  # Color of band lines
		    'bands_linewidth',  # linewidth of bands
		    'bands_linestyle',  # linestyle of bands
		    'bands_marker',  # marker for bands
		    'bands_markersize',  # size of the marker of bands
		    'bands_markeredgecolor',  # marker edge color for bands
		    'bands_markeredgewidth',  # marker edge width for bands
		    'bands_markerfacecolor',  # marker face color for bands
		    'use_latex',  # If true, use latex to render captions
		]

		# Note: I do not want to import matplotlib here, for two reasons:
		# 1. I would like to be able to print the script for the user
		# 2. I don't want to mess up with the user matplotlib backend
		#    (that I should do if the user does not have a X server, but that
		#    I do not want to do if he's e.g. in jupyter)
		# Therefore I just create a string that can be executed as needed, e.g. with eval.
		# I take care of sanitizing the output.
		# if prettify_format is None:
		#     # Default. Specified like this to allow caller functions to pass 'None'
		#     prettify_format = 'latex_seekpath'

		# # The default for use_latex is False
		# join_symbol = r'\textbar{}' if kwargs.get('use_latex', False) else '|'

		plot_info = self._get_stabilityplot_data()

		all_data = {}

		all_data['x'] = plot_info['x'].tolist()
		all_data['compound_lines'] = plot_info['y'].tolist()
		all_data['stability_vertices'] = plot_info['vertices'].tolist()
		all_data['boundary_lines'] = list(plot_info['boundary_lines'])
		all_data['all_compounds'] = self.index
		if self.property_map:
			all_data['grid'] = plot_info['grid']
			all_data['property'] = plot_info['property']
			# all_data['grid_dx'] = (np.amax(plot_info['vertices'][:, 0]) - np.amin(plot_info['vertices'][:, 0]))/50
			# all_data['grid_dy'] = (np.amax(plot_info['vertices'][:, 1]) - np.amin(plot_info['vertices'][:, 1]))/50
		all_data['legend_location'] = legend_location
		all_data['xaxis_label'] = f'Chemical potential of {self.column[0]} (eV)'
		all_data['yaxis_label'] = f'Chemical potential of {self.column[1]} (eV)'
		all_data['title'] = title
		# if comments:
		#     all_data['comment'] = prepare_header_comment(self.uuid, plot_info, comment_char='#')

		# axis limits
		width = np.amax(plot_info['vertices'][:,0]) - np.amin(plot_info['vertices'][:,0]) # width of the stability region
		height = np.amax(plot_info['vertices'][:,1]) - np.amin(plot_info['vertices'][:,1]) # height of the stability region
		if y_max_lim is None:
		    y_max_lim = min(0, np.amax(plot_info['vertices'][:,1])+0.2*height)
		if y_min_lim is None:
		    y_min_lim = np.amin(plot_info['vertices'][:,1])-0.2*height
		if x_max_lim is None:
			x_max_lim = min(0, np.amax(plot_info['vertices'][:,0])+0.2*width)
		if x_min_lim is None:
			x_min_lim = np.amin(plot_info['vertices'][:,0])-0.2*width
		all_data['x_min_lim'] = x_min_lim
		all_data['x_max_lim'] = x_max_lim
		all_data['y_min_lim'] = y_min_lim
		all_data['y_max_lim'] = y_max_lim

		for key, value in kwargs.items():
		    if key not in valid_additional_keywords:
		        raise TypeError(f"_matplotlib_get_dict() got an unexpected keyword argument '{key}'")
		    all_data[key] = value

		return all_data

	def _prepare_mpl_singlefile(self, *args, **kwargs):
		"""
		Prepare a python script using matplotlib to plot the bands

		For the possible parameters, see documentation of
		:py:meth:`~aiida.orm.nodes.data.array.bands.BandsData._matplotlib_get_dict`
		"""
		from aiida.common import json

		all_data = self._matplotlib_get_dict(*args, **kwargs)

		s_header = MATPLOTLIB_HEADER_TEMPLATE.substitute()
		s_import = MATPLOTLIB_IMPORT_DATA_INLINE_TEMPLATE.substitute(all_data_json=json.dumps(all_data, indent=2))
		s_body = self._get_mpl_body_template(all_data)
		# s_body = MATPLOTLIB_BODY_TEMPLATE.substitute()
		s_footer = MATPLOTLIB_FOOTER_TEMPLATE_SHOW.substitute()

		string = s_header + s_import + s_body + s_footer

		return string.encode('utf-8'), {}

	def _prepare_mpl_pdf(self, main_file_name='', *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg,unused-argument
		"""
		Prepare a python script using matplotlib to plot the stability region, with the JSON
		returned as an independent file.
		"""
		import os
		import tempfile
		import subprocess
		import sys

		from aiida.common import json

		all_data = self._matplotlib_get_dict(*args, **kwargs)

		# Use the Agg backend
		s_header = MATPLOTLIB_HEADER_AGG_TEMPLATE.substitute()
		s_import = MATPLOTLIB_IMPORT_DATA_INLINE_TEMPLATE.substitute(all_data_json=json.dumps(all_data, indent=2))
		s_body = self._get_mpl_body_template(all_data)

		# I get a temporary file name
		handle, filename = tempfile.mkstemp()
		os.close(handle)
		os.remove(filename)

		escaped_fname = filename.replace('"', '\"')

		s_footer = MATPLOTLIB_FOOTER_TEMPLATE_EXPORTFILE.substitute(fname=escaped_fname, format='pdf')

		string = s_header + s_import + s_body + s_footer

		# I don't exec it because I might mess up with the matplotlib backend etc.
		# I run instead in a different process, with the same executable
		# (so it should work properly with virtualenvs)
		# with tempfile.NamedTemporaryFile(mode='w+') as handle:
		#     handle.write(string)
		#     handle.flush()
		#     subprocess.check_output([sys.executable, handle.name])

		if not os.path.exists(filename):
		    raise RuntimeError('Unable to generate the PDF...')

		with open(filename, 'rb', encoding=None) as handle:
		    imgdata = handle.read()
		os.remove(filename)

		return imgdata, {}

	def _prepare_mpl_withjson(self, main_file_name='', *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
		"""
		Prepare a python script using matplotlib to plot the bands, with the JSON
		returned as an independent file.

		For the possible parameters, see documentation of
		:py:meth:`~aiida.orm.nodes.data.array.bands.BandsData._matplotlib_get_dict`
		"""
		import os

		from aiida.common import json

		all_data = self._matplotlib_get_dict(*args, main_file_name=main_file_name, **kwargs)

		json_fname = os.path.splitext(main_file_name)[0] + '_data.json'
		# Escape double_quotes
		json_fname = json_fname.replace('"', '\"')

		ext_files = {json_fname: json.dumps(all_data, indent=2).encode('utf-8')}

		s_header = MATPLOTLIB_HEADER_TEMPLATE.substitute()
		s_import = MATPLOTLIB_IMPORT_DATA_FROMFILE_TEMPLATE.substitute(json_fname=json_fname)
		s_body   = self._get_mpl_body_template(all_data)
		s_footer = MATPLOTLIB_FOOTER_TEMPLATE_SHOW.substitute()

		string = s_header + s_import + s_body + s_footer

		return string.encode('utf-8'), ext_files

	@staticmethod
	def _get_mpl_body_template(all_data):
	    if all_data.get('grid'):
	        s_body = MATPLOTLIB_BODY_TEMPLATE.substitute(plot_code=WITH_PROPERTY)
	    else:
	        s_body = MATPLOTLIB_BODY_TEMPLATE.substitute(plot_code=WITHOUT_PROPERTY)
	    return s_body

	def show_mpl(self, **kwargs):
		"""
		Call a show() command for the band structure using matplotlib.
		This uses internally the 'mpl_singlefile' format, with empty
		main_file_name.

		Other kwargs are passed to self._exportcontent.
		"""
		exec(*self._exportcontent(fileformat='mpl_singlefile', main_file_name='', **kwargs))

	def _prepare_json(self, main_file_name='', comments=True):  # pylint: disable=unused-argument
		"""
		Prepare a json file in a format compatible with the AiiDA band visualizer

		:param comments: if True, print comments (if it makes sense for the given
		    format)
		"""
		from aiida import get_file_header
		from aiida.common import json

		json_dict = self._get_band_segments(cartesian=True)
		json_dict['original_uuid'] = self.uuid

		if comments:
		    json_dict['comments'] = get_file_header(comment_char='')

		return json.dumps(json_dict).encode('utf-8'), {}

MATPLOTLIB_HEADER_AGG_TEMPLATE = Template(
    """# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

from matplotlib import rc
# Uncomment to change default font
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern', 'CMU Serif', 'Times New Roman', 'DejaVu Serif']})
# To use proper font for, e.g., Gamma if usetex is set to False
rc('mathtext', fontset='cm')

rc('text', usetex=True)

import pylab as pl

# I use json to make sure the input is sanitized
import json

print_comment = False
"""
)

MATPLOTLIB_HEADER_TEMPLATE = Template(
    """# -*- coding: utf-8 -*-

from matplotlib import rc
# Uncomment to change default font
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern', 'CMU Serif', 'Times New Roman', 'DejaVu Serif']})
# To use proper font for, e.g., Gamma if usetex is set to False
rc('mathtext', fontset='cm')

rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np

# I use json to make sure the input is sanitized
import json

print_comment = False

def prettify_compound_name(name):
	pretty_name = ''
	for char in name:
		if char.isnumeric():
			pretty_name += '$$_'+char+'$$'
		else:
			pretty_name += char
	return pretty_name

"""
)

MATPLOTLIB_IMPORT_DATA_INLINE_TEMPLATE = Template('''all_data_str = r"""$all_data_json"""
''')

MATPLOTLIB_IMPORT_DATA_FROMFILE_TEMPLATE = Template(
    """with open("$json_fname", encoding='utf8') as f:
    all_data_str = f.read()
"""
)

WITHOUT_PROPERTY = '''
ax.fill(vertices[:,0], vertices[:,1], color='gray', alpha=0.5)
'''

WITH_PROPERTY = '''
from scipy.interpolate import griddata
import matplotlib.colors as colors
import matplotlib.cm as cm

x_min = np.amin(vertices[:,0])
x_max = np.amax(vertices[:,0])
y_min = np.amin(vertices[:,1])
y_max = np.amax(vertices[:,1])
num_point = 100
X = np.linspace(x_min, x_max, num_point)
Y = np.linspace(y_min, y_max, num_point)
xx, yy = np.meshgrid(X, Y)

interp_c = griddata(all_data['grid'], all_data['property'], (xx, yy), method='linear')
im = ax.pcolor(X, Y, interp_c, norm=colors.LogNorm(vmin=np.nanmin(interp_c)*0.95, vmax=np.nanmax(interp_c)*0.95*1.05), cmap=cm.RdBu, shading='auto')
cbar = fig.colorbar(im, ax=ax, extend='max')
cbar.ax.set_ylabel('Concentration (/cm$^3$)', fontsize=14, rotation=-90, va="bottom")
'''


MATPLOTLIB_BODY_TEMPLATE = Template(
    """all_data = json.loads(all_data_str)

if not all_data.get('use_latex', False):
    rc('text', usetex=False)

# Option for bands (all, or those of type 1 if there are two spins)
further_plot_options = {}
further_plot_options['color'] = all_data.get('bands_color', 'k')
further_plot_options['linewidth'] = all_data.get('bands_linewidth', 0.5)
further_plot_options['linestyle'] = all_data.get('bands_linestyle', None)
further_plot_options['marker'] = all_data.get('bands_marker', None)
further_plot_options['markersize'] = all_data.get('bands_markersize', None)
further_plot_options['markeredgecolor'] = all_data.get('bands_markeredgecolor', None)
further_plot_options['markeredgewidth'] = all_data.get('bands_markeredgewidth', None)
further_plot_options['markerfacecolor'] = all_data.get('bands_markerfacecolor', None)

fig, ax = plt.subplots()

for h, v in zip(all_data['x'], all_data['compound_lines']):
	ax.plot(h, v, linestyle='dashed', linewidth=1.0, color='k')

for cmp in all_data['boundary_lines']:
	idx = all_data['all_compounds'].index(cmp)
	ax.plot(all_data['x'][idx], all_data['compound_lines'][idx], linewidth=1.5, label=prettify_compound_name(cmp))

vertices = np.array(all_data['stability_vertices'])
ax.scatter(vertices[:,0], vertices[:,1], color='k')

${plot_code}

ax.set_xlim([all_data['x_min_lim'], all_data['x_max_lim']])
ax.set_ylim([all_data['y_min_lim'], all_data['y_max_lim']])
# p.xaxis.grid(True, which='major', color='#888888', linestyle='-', linewidth=0.5)

if all_data['title']:
    ax.set_title(all_data['title'])
if all_data['legend_location']:
	ax.legend(loc=all_data['legend_location'])
ax.set_xlabel(all_data['xaxis_label'])
ax.set_ylabel(all_data['yaxis_label'])

try:
    if print_comment:
        print(all_data['comment'])
except KeyError:
    pass
"""
)

MATPLOTLIB_FOOTER_TEMPLATE_SHOW = Template("""plt.show()""")

MATPLOTLIB_FOOTER_TEMPLATE_EXPORTFILE = Template("""plt.savefig("$fname", format="$format")""")

MATPLOTLIB_FOOTER_TEMPLATE_EXPORTFILE_WITH_DPI = Template("""plt.savefig("$fname", format="$format", dpi=$dpi)""")