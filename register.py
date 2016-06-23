#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-23
import os
import pypandoc

f = open("./README.txt", "w+")
f.write(pypandoc.convert("./README.md", "rst"))
f.close()
os.system("python setup.py sdist upload")
os.remove("README.txt")
