import os.path
release = False
__version__ = '0.1'


_repository_path = os.path.split(__file__)[0]
_git_file_path = os.path.join(_repository_path, '__git_version__.py')

def get_git_revision():
    '''
    Gets the last GIT commit hash and date for the repository, using the
    path to this file.
    '''
    from subprocess import Popen, PIPE
    rev = ''
    try:
        proc = Popen(['git', 'log', '-1', '--date=short', '--format="%h %ad"', 
                      _repository_path], stdout=PIPE)
        text = proc.stdout.readline()
        text = text.replace('"', '').split()
        rev = '.dev.' + text[0] + '.' + text[1]
    except OSError:
        # GIT not installed, don't worry about finding the revision hash
        pass
    return rev

def write_git_version():
    'Write the GIT revision to a file.'
    gitfile = open(_git_file_path, 'w')
    gitfile.write('rev = "%s"\n' % get_git_revision())
    gitfile.close()


def get_version():
    '''
    Get the version of the package, including the GIT revision if this
    is an actual release.
    '''
    version = __version__
    if not release:
        try:
            import __git_version__
            version += __git_version__.rev
        except ImportError:
            version += get_git_revision()
            pass
    return version
