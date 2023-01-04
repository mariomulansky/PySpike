## interlude to force answer to input('Abort?'):

import io, sys
sys.stdin = io.StringIO('N\n')
import setup
