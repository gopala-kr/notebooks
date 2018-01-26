# Copyright 2015 The deep-ml.com Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
sys.path.append('/root/inst/bin/')

# print (os.environ["PATH"])

from distutils.spawn import find_executable
whichCling = find_executable('cling')

print ('cling =' + str(whichCling))

c.NotebookApp.ip = '*'
c.NotebookApp.port = 7842
c.NotebookApp.open_browser = False
c.MultiKernelManager.default_kernel_name = 'python2'

# sets a password if PASSWORD is set in the environment
c.NotebookApp.token = ''
#c.NotebookApp.password = ''
# password is eric=123
c.NotebookApp.password='sha1:65db47cf7e0d:d440485d58ec9fcc8b587c0aa96864f2f1816edd'
