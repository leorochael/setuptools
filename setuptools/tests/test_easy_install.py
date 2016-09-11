# -*- coding: utf-8 -*-
"""Easy install Tests
"""
from __future__ import absolute_import

import sys
import os
import os.path
import tempfile
import site
import contextlib
import tarfile
import logging
import itertools
import distutils.errors
import io
import subprocess
import zipfile
from unittest import mock

import time
from six.moves import urllib

import pytest

from setuptools import sandbox
from setuptools.sandbox import run_setup
import setuptools.command.easy_install as ei
from setuptools.command.easy_install import PthDistributions
from setuptools.command import easy_install as easy_install_pkg
from setuptools.dist import Distribution
from pkg_resources import normalize_path, working_set
from pkg_resources import Distribution as PRDistribution
from pkg_resources import PY_MAJOR
import setuptools.tests.server
import pkg_resources

from .py26compat import tarfile_open
from . import contexts
from .textwrap import DALS


class FakeDist(object):
    def get_entry_map(self, group):
        if group != 'console_scripts':
            return {}
        return {'name': 'ep'}

    def as_requirement(self):
        return 'spec'


SETUP_PY = DALS("""
    from setuptools import setup

    setup(name='foo')
    """)


class TestEasyInstallTest:
    def test_install_site_py(self, tmpdir):
        dist = Distribution()
        cmd = ei.easy_install(dist)
        cmd.sitepy_installed = False
        cmd.install_dir = str(tmpdir)
        cmd.install_site_py()
        assert (tmpdir / 'site.py').exists()

    def test_get_script_args(self):
        header = ei.CommandSpec.best().from_environment().as_header()
        expected = header + DALS(r"""
            # EASY-INSTALL-ENTRY-SCRIPT: 'spec','console_scripts','name'
            __requires__ = 'spec'
            import re
            import sys
            from pkg_resources import load_entry_point

            if __name__ == '__main__':
                sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
                sys.exit(
                    load_entry_point('spec', 'console_scripts', 'name')()
                )
            """)
        dist = FakeDist()

        args = next(ei.ScriptWriter.get_args(dist))
        name, script = itertools.islice(args, 2)

        assert script == expected

    def test_no_find_links(self):
        # new option '--no-find-links', that blocks find-links added at
        # the project level
        dist = Distribution()
        cmd = ei.easy_install(dist)
        cmd.check_pth_processing = lambda: True
        cmd.no_find_links = True
        cmd.find_links = ['link1', 'link2']
        cmd.install_dir = os.path.join(tempfile.mkdtemp(), 'ok')
        cmd.args = ['ok']
        cmd.ensure_finalized()
        assert cmd.package_index.scanned_urls == {}

        # let's try without it (default behavior)
        cmd = ei.easy_install(dist)
        cmd.check_pth_processing = lambda: True
        cmd.find_links = ['link1', 'link2']
        cmd.install_dir = os.path.join(tempfile.mkdtemp(), 'ok')
        cmd.args = ['ok']
        cmd.ensure_finalized()
        keys = sorted(cmd.package_index.scanned_urls.keys())
        assert keys == ['link1', 'link2']

    def test_write_exception(self):
        """
        Test that `cant_write_to_target` is rendered as a DistutilsError.
        """
        dist = Distribution()
        cmd = ei.easy_install(dist)
        cmd.install_dir = os.getcwd()
        with pytest.raises(distutils.errors.DistutilsError):
            cmd.cant_write_to_target()

    def test_all_site_dirs(self, monkeypatch):
        """
        get_site_dirs should always return site dirs reported by
        site.getsitepackages.
        """
        path = normalize_path('/setuptools/test/site-packages')
        mock_gsp = lambda: [path]
        monkeypatch.setattr(site, 'getsitepackages', mock_gsp, raising=False)
        assert path in ei.get_site_dirs()

    def test_all_site_dirs_works_without_getsitepackages(self, monkeypatch):
        monkeypatch.delattr(site, 'getsitepackages', raising=False)
        assert ei.get_site_dirs()

    @pytest.fixture
    def sdist_unicode(self, tmpdir):
        files = [
            (
                'setup.py',
                DALS("""
                    import setuptools
                    setuptools.setup(
                        name="setuptools-test-unicode",
                        version="1.0",
                        packages=["mypkg"],
                        include_package_data=True,
                    )
                    """),
            ),
            (
                'mypkg/__init__.py',
                "",
            ),
            (
                u'mypkg/\u2603.txt',
                "",
            ),
        ]
        sdist_name = 'setuptools-test-unicode-1.0.zip'
        sdist = str(tmpdir / sdist_name)
        # can't use make_tgz, because the issue only occurs
        #  with zip sdists.
        make_zip(sdist, files)
        return sdist

    def test_unicode_filename_in_sdist(self, sdist_unicode, tmpdir, monkeypatch):
        """
        The install command should execute correctly even if
        the package has unicode filenames.
        """
        dist = Distribution({'script_args': ['easy_install']})
        target = (tmpdir / 'target').ensure_dir()
        cmd = ei.easy_install(
            dist,
            install_dir=str(target),
            args=['x'],
        )
        monkeypatch.setitem(os.environ, 'PYTHONPATH', str(target))
        cmd.ensure_finalized()
        cmd.easy_install(sdist_unicode)


class TestPTHFileWriter:
    def test_add_from_cwd_site_sets_dirty(self):
        '''a pth file manager should set dirty
        if a distribution is in site but also the cwd
        '''
        pth = PthDistributions('does-not_exist', [os.getcwd()])
        assert not pth.dirty
        pth.add(PRDistribution(os.getcwd()))
        assert pth.dirty

    def test_add_from_site_is_ignored(self):
        location = '/test/location/does-not-have-to-exist'
        # PthDistributions expects all locations to be normalized
        location = pkg_resources.normalize_path(location)
        pth = PthDistributions('does-not_exist', [location, ])
        assert not pth.dirty
        pth.add(PRDistribution(location))
        assert not pth.dirty


@pytest.yield_fixture
def setup_context(tmpdir):
    with (tmpdir / 'setup.py').open('w') as f:
        f.write(SETUP_PY)
    with tmpdir.as_cwd():
        yield tmpdir


@pytest.mark.usefixtures("user_override")
@pytest.mark.usefixtures("setup_context")
class TestUserInstallTest:

    # prevent check that site-packages is writable. easy_install
    # shouldn't be writing to system site-packages during finalize
    # options, but while it does, bypass the behavior.
    prev_sp_write = mock.patch(
        'setuptools.command.easy_install.easy_install.check_site_dir',
        mock.Mock(),
    )

    # simulate setuptools installed in user site packages
    @mock.patch('setuptools.command.easy_install.__file__', site.USER_SITE)
    @mock.patch('site.ENABLE_USER_SITE', True)
    @prev_sp_write
    def test_user_install_not_implied_user_site_enabled(self):
        self.assert_not_user_site()

    @mock.patch('site.ENABLE_USER_SITE', False)
    @prev_sp_write
    def test_user_install_not_implied_user_site_disabled(self):
        self.assert_not_user_site()

    @staticmethod
    def assert_not_user_site():
        # create a finalized easy_install command
        dist = Distribution()
        dist.script_name = 'setup.py'
        cmd = ei.easy_install(dist)
        cmd.args = ['py']
        cmd.ensure_finalized()
        assert not cmd.user, 'user should not be implied'

    def test_multiproc_atexit(self):
        pytest.importorskip('multiprocessing')

        log = logging.getLogger('test_easy_install')
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        log.info('this should not break')

    @pytest.fixture()
    def foo_package(self, tmpdir):
        egg_file = tmpdir / 'foo-1.0.egg-info'
        with egg_file.open('w') as f:
            f.write('Name: foo\n')
        return str(tmpdir)

    @pytest.yield_fixture()
    def install_target(self, tmpdir):
        target = str(tmpdir)
        with mock.patch('sys.path', sys.path + [target]):
            python_path = os.path.pathsep.join(sys.path)
            with mock.patch.dict(os.environ, PYTHONPATH=python_path):
                yield target

    def test_local_index(self, foo_package, install_target):
        """
        The local index must be used when easy_install locates installed
        packages.
        """
        dist = Distribution()
        dist.script_name = 'setup.py'
        cmd = ei.easy_install(dist)
        cmd.install_dir = install_target
        cmd.args = ['foo']
        cmd.ensure_finalized()
        cmd.local_index.scan([foo_package])
        res = cmd.easy_install('foo')
        actual = os.path.normcase(os.path.realpath(res.location))
        expected = os.path.normcase(os.path.realpath(foo_package))
        assert actual == expected

    @contextlib.contextmanager
    def user_install_setup_context(self, *args, **kwargs):
        """
        Wrap sandbox.setup_context to patch easy_install in that context to
        appear as user-installed.
        """
        with self.orig_context(*args, **kwargs):
            import setuptools.command.easy_install as ei
            ei.__file__ = site.USER_SITE
            yield

    def patched_setup_context(self):
        self.orig_context = sandbox.setup_context

        return mock.patch(
            'setuptools.sandbox.setup_context',
            self.user_install_setup_context,
        )

@pytest.mark.usefixtures("user_override")
class TestEasyInstallMainToDirectory:

    dist_name = "myorg.mypackage"
    dist_version = "1.0"
    dist_name_version = "%s-%s" % (dist_name, dist_version)
    egg_basename = "%s-py%s.egg" % (dist_name_version, PY_MAJOR)
    egg_platform_basename = "%s-py%s-%s.egg" % (
        dist_name_version, PY_MAJOR, pkg_resources.get_build_platform(),
    )

    def sdist(self, tmpdir, suffix, method):
        src_dir = tmpdir.mkdir("src")
        sdist = tmpdir / (self.dist_name_version + suffix)

        make_nspkg_sdist(
            str(sdist),
            self.dist_name,
            self.dist_version,
            method=method,
        )
        return sdist

    @pytest.fixture
    def sdist_tgz(self, tmpdir):
        return self.sdist(tmpdir, suffix=".tar.gz", method=make_tgz)

    @pytest.fixture
    def sdist_zip(self, tmpdir):
        return self.sdist(tmpdir, suffix=".zip", method=make_zip)

    @pytest.fixture
    def sdist_dir(self, tmpdir):
        return self.sdist(tmpdir, suffix="", method=make_tree)

    @pytest.fixture
    def dist_setup(self, sdist_dir):
        from distutils.core import run_setup as du_run_setup

        def dist_setup(dist_cmd, stop_after="run", run_after=lambda dist: None):
            with sdist_dir.as_cwd():
                dist = du_run_setup(
                    str(sdist_dir/"setup.py"),
                    [dist_cmd],
                )
                run_after(dist)
                [bdist] = [
                    sdist_dir / filename
                    for cmd, pyversion, filename in dist.dist_files
                    if cmd == dist_cmd
                ]
            return bdist

        return dist_setup

    @pytest.fixture
    def index_dir(self, tmpdir):
        return tmpdir.mkdir("index")

    @pytest.fixture
    def index_dist_page(self, index_dir):
        return index_dir.mkdir(self.dist_name)

    @pytest.fixture
    def check(self, index_dir, tmpdir):
        install_dir = (tmpdir/"eggs")
        install_dir.mkdir()
        argv = [
            '-mZNxqq',
            '-i', str(index_dir),
            '-d', str(install_dir),
        ]

        def check():
            pi = setuptools.package_index.PackageIndex(str(index_dir))
            requirement = pkg_resources.Requirement.parse(self.dist_name)
            pi.find_packages(requirement)
            assert len(pi[self.dist_name]) == 1, pi

            ei.main(argv + [self.dist_name])
            expected_locations = set([
                install_dir/self.egg_basename,
                install_dir/self.egg_platform_basename,
            ])
            [expected_location] = expected_locations.intersection(
                install_dir.listdir())
            pkg_env = pkg_resources.Environment([str(expected_location)])
            [installed_dist] = pkg_env[self.dist_name]
            assert installed_dist.version == self.dist_version

        return check

    def test_sdist_tgz_install(self, index_dist_page, check, sdist_tgz):
        # install from source tgz on index
        sdist_tgz.copy(index_dist_page)

        check()

    def test_sdist_zip_install(self, index_dist_page, check, sdist_zip):
        # install from source zip on index
        sdist_zip.copy(index_dist_page)

        check()

    def test_egg_install(self, index_dist_page, check, dist_setup):
        # install from egg on index:
        dist_filename = dist_setup("bdist_egg")

        index_dist_egg = index_dist_page/self.egg_basename
        dist_filename.copy(index_dist_page)
        assert zipfile.is_zipfile(str(index_dist_egg))

        check()

    def test_wininst_install(self, index_dist_page, check, dist_setup,
                             monkeypatch):
        # build wininst:
        from setuptools.command.bdist_wininst import bdist_wininst

        class bdist_wininst_patched(bdist_wininst):
            def get_exe_bytes(self):
                """We are not going to execute the installer wrapper"""
                return b''
            def get_inidata(self):
                """Avoid attempt by create_exe() to encode cfgdata as 'mbcs'
                which doesn't exist on nixes"""
                cfgdata = bdist_wininst.get_inidata(self)
                if not isinstance(cfgdata, bytes):
                    cfgdata = cfgdata.encode('ascii')
                return cfgdata

        def run_wininst(dist):
            cmd = bdist_wininst_patched(dist)
            cmd.target_version = PY_MAJOR
            cmd.ensure_finalized()
            cmd.run()

        dist_filename = dist_setup(
            "bdist_wininst",
            stop_after="cmdline",
            run_after=run_wininst,
        )

        # copy wininst to index:
        index_dist_wininst = index_dist_page/dist_filename.basename
        dist_filename.copy(index_dist_wininst)
        assert zipfile.is_zipfile(str(index_dist_wininst))

        def parse_bdist_wininst(name):
            """The original checks against specific platforms like:
                win-amd64 and win32, instead of parsing the platform from the
                file, but the fake wininst will match the current platform
            """
            name, _ = os.path.splitext(name)
            base_plat, py_ver = name.rsplit("-", 1)
            assert py_ver.startswith("py")
            py_ver = py_ver[2:]
            base, plat = base_plat.rsplit(".", 1)
            return base, py_ver, plat

        monkeypatch.setattr(
            'setuptools.package_index.parse_bdist_wininst',
            parse_bdist_wininst,
        )
        check()

    @mock.patch('distutils.core.DEBUG', True)
    def test_wheel_install(self, index_dist_page, check, sdist_dir):
        # build wheel:
        from distutils.core import run_setup as du_run_setup
        from wheel.bdist_wheel import bdist_wheel
        with sdist_dir.as_cwd():
            dist = du_run_setup(
                str(sdist_dir/"setup.py"),
                ["build"],
                "commandline",
            )
            cmd = bdist_wheel(dist)
            cmd.universal = True
            cmd.ensure_finalized()
            cmd.run()
        # copy wheel to index:
        wheel_basename = self.dist_name_version + "-py2.py3-none-any.whl"
        index_dist_wheel = index_dist_page/wheel_basename
        (sdist_dir/"dist"/wheel_basename).copy(index_dist_page)
        assert zipfile.is_zipfile(str(index_dist_page/wheel_basename))

        check()

@pytest.yield_fixture
def distutils_package():
    distutils_setup_py = SETUP_PY.replace(
        'from setuptools import setup',
        'from distutils.core import setup',
    )
    with contexts.tempdir(cd=os.chdir):
        with open('setup.py', 'w') as f:
            f.write(distutils_setup_py)
        yield


class TestDistutilsPackage:
    def test_bdist_egg_available_on_distutils_pkg(self, distutils_package):
        run_setup('setup.py', ['bdist_egg'])


class TestSetupRequires:
    def test_setup_requires_honors_fetch_params(self):
        """
        When easy_install installs a source distribution which specifies
        setup_requires, it should honor the fetch parameters (such as
        allow-hosts, index-url, and find-links).
        """
        # set up a server which will simulate an alternate package index.
        p_index = setuptools.tests.server.MockServer()
        p_index.start()
        netloc = 1
        p_index_loc = urllib.parse.urlparse(p_index.url)[netloc]
        if p_index_loc.endswith(':0'):
            # Some platforms (Jython) don't find a port to which to bind,
            #  so skip this test for them.
            return
        with contexts.quiet():
            # create an sdist that has a build-time dependency.
            with TestSetupRequires.create_sdist() as dist_file:
                with contexts.tempdir() as temp_install_dir:
                    with contexts.environment(PYTHONPATH=temp_install_dir):
                        ei_params = [
                            '--index-url', p_index.url,
                            '--allow-hosts', p_index_loc,
                            '--exclude-scripts',
                            '--install-dir', temp_install_dir,
                            dist_file,
                        ]
                        with sandbox.save_argv(['easy_install']):
                            # attempt to install the dist. It should fail because
                            #  it doesn't exist.
                            with pytest.raises(SystemExit):
                                easy_install_pkg.main(ei_params)
        # there should have been two or three requests to the server
        #  (three happens on Python 3.3a)
        assert 2 <= len(p_index.requests) <= 3
        assert p_index.requests[0].path == '/does-not-exist/'

    @staticmethod
    @contextlib.contextmanager
    def create_sdist():
        """
        Return an sdist with a setup_requires dependency (of something that
        doesn't exist)
        """
        with contexts.tempdir() as dir:
            dist_path = os.path.join(dir, 'setuptools-test-fetcher-1.0.tar.gz')
            make_tgz(dist_path, [
                ('setup.py', DALS("""
                    import setuptools
                    setuptools.setup(
                        name="setuptools-test-fetcher",
                        version="1.0",
                        setup_requires = ['does-not-exist'],
                    )
                """))])
            yield dist_path

    def test_setup_requires_overrides_version_conflict(self):
        """
        Regression test for distribution issue 323:
        https://bitbucket.org/tarek/distribute/issues/323

        Ensures that a distribution's setup_requires requirements can still be
        installed and used locally even if a conflicting version of that
        requirement is already on the path.
        """

        fake_dist = PRDistribution('does-not-matter', project_name='foobar',
                                   version='0.0')
        working_set.add(fake_dist)

        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                test_pkg = create_setup_requires_package(temp_dir)
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                with contexts.quiet() as (stdout, stderr):
                    # Don't even need to install the package, just
                    # running the setup.py at all is sufficient
                    run_setup(test_setup_py, ['--name'])

                lines = stdout.readlines()
                assert len(lines) > 0
                assert lines[-1].strip(), 'test_pkg'

    def test_setup_requires_override_nspkg(self):
        """
        Like ``test_setup_requires_overrides_version_conflict`` but where the
        ``setup_requires`` package is part of a namespace package that has
        *already* been imported.
        """

        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                foobar_1_archive = os.path.join(temp_dir, 'foo.bar-0.1.tar.gz')
                make_nspkg_sdist(foobar_1_archive, 'foo.bar', '0.1')
                # Now actually go ahead an extract to the temp dir and add the
                # extracted path to sys.path so foo.bar v0.1 is importable
                foobar_1_dir = os.path.join(temp_dir, 'foo.bar-0.1')
                os.mkdir(foobar_1_dir)
                with tarfile_open(foobar_1_archive) as tf:
                    tf.extractall(foobar_1_dir)
                sys.path.insert(1, foobar_1_dir)

                dist = PRDistribution(foobar_1_dir, project_name='foo.bar',
                                      version='0.1')
                working_set.add(dist)

                template = DALS("""\
                    import foo  # Even with foo imported first the
                                # setup_requires package should override
                    import setuptools
                    setuptools.setup(**%r)

                    if not (hasattr(foo, '__path__') and
                            len(foo.__path__) == 2):
                        print('FAIL')

                    if 'foo.bar-0.2' not in foo.__path__[0]:
                        print('FAIL')
                """)

                test_pkg = create_setup_requires_package(
                    temp_dir, 'foo.bar', '0.2', make_nspkg_sdist, template)

                test_setup_py = os.path.join(test_pkg, 'setup.py')

                with contexts.quiet() as (stdout, stderr):
                    try:
                        # Don't even need to install the package, just
                        # running the setup.py at all is sufficient
                        run_setup(test_setup_py, ['--name'])
                    except pkg_resources.VersionConflict:
                        self.fail('Installing setup.py requirements '
                            'caused a VersionConflict')

                assert 'FAIL' not in stdout.getvalue()
                lines = stdout.readlines()
                assert len(lines) > 0
                assert lines[-1].strip() == 'test_pkg'


def make_tgz(dist_path, files):
    """
    Create a simple sdist tarball at dist_path, containing the files
    listed in ``files`` as ``(filename, content)`` tuples.
    """

    with tarfile_open(dist_path, 'w:gz') as dist:
        for filename, content in files:
            file_bytes = io.BytesIO(content.encode('utf-8'))
            file_info = tarfile.TarInfo(name=filename)
            file_info.size = len(file_bytes.getvalue())
            file_info.mtime = int(time.time())
            dist.addfile(file_info, fileobj=file_bytes)


def make_zip(zip_path, files):
    """
    Create a simple zip file at zip_path, containing the files
    listed in ``files`` as ``(filename, content)`` tuples.
    """
    with zipfile.ZipFile(zip_path, 'w') as zip_package:
        for filename, content in files:
            zip_package.writestr(filename, content)
        zip_package.close()


def make_tree(path, files):
    """
    Create a directory tree, containing the files
    listed in ``files`` as ``(filename, content)`` tuples.
    """
    os.mkdir(path)
    for filename, content in files:
        fullpath = os.path.join(path, filename)
        dirpath = os.path.dirname(fullpath)
        if not os.path.isdir(dirpath):
            os.makedirs(os.path.dirname(fullpath))
        with open(fullpath, "wb") as f:
            f.write(content.encode('utf-8'))


def make_trivial_sdist(dist_path, distname, version, method=make_tgz):
    """
    Create a simple sdist tarball at dist_path, containing just a simple
    setup.py.
    """

    method(dist_path, [
        ('setup.py',
         DALS("""\
             import setuptools
             setuptools.setup(
                 name=%r,
                 version=%r
             )
         """ % (distname, version)))])


def make_nspkg_sdist(dist_path, distname, version, method=make_tgz):
    """
    Make an sdist tarball with distname and version which also contains one
    package with the same name as distname.  The top-level package is
    designated a namespace package).
    """

    parts = distname.split('.')
    nspackage = parts[0]

    packages = ['.'.join(parts[:idx]) for idx in range(1, len(parts) + 1)]

    setup_py = DALS("""\
        import setuptools
        setuptools.setup(
            name=%r,
            version=%r,
            packages=%r,
            namespace_packages=[%r]
        )
    """ % (distname, version, packages, nspackage))

    init = "__import__('pkg_resources').declare_namespace(__name__)"

    files = [('setup.py', setup_py),
             (os.path.join(nspackage, '__init__.py'), init)]
    for package in packages[1:]:
        filename = os.path.join(*(package.split('.') + ['__init__.py']))
        files.append((filename, ''))

    method(dist_path, files)


def create_setup_requires_package(path, distname='foobar', version='0.1',
                                  make_package=make_trivial_sdist,
                                  setup_py_template=None):
    """Creates a source tree under path for a trivial test package that has a
    single requirement in setup_requires--a tarball for that requirement is
    also created and added to the dependency_links argument.

    ``distname`` and ``version`` refer to the name/version of the package that
    the test package requires via ``setup_requires``.  The name of the test
    package itself is just 'test_pkg'.
    """

    test_setup_attrs = {
        'name': 'test_pkg', 'version': '0.0',
        'setup_requires': ['%s==%s' % (distname, version)],
        'dependency_links': [os.path.abspath(path)]
    }

    test_pkg = os.path.join(path, 'test_pkg')
    test_setup_py = os.path.join(test_pkg, 'setup.py')
    os.mkdir(test_pkg)

    if setup_py_template is None:
        setup_py_template = DALS("""\
            import setuptools
            setuptools.setup(**%r)
        """)

    with open(test_setup_py, 'w') as f:
        f.write(setup_py_template % test_setup_attrs)

    foobar_path = os.path.join(path, '%s-%s.tar.gz' % (distname, version))
    make_package(foobar_path, distname, version)

    return test_pkg


@pytest.mark.skipif(
    sys.platform.startswith('java') and ei.is_sh(sys.executable),
    reason="Test cannot run under java when executable is sh"
)
class TestScriptHeader:
    non_ascii_exe = '/Users/José/bin/python'
    exe_with_spaces = r'C:\Program Files\Python33\python.exe'

    def test_get_script_header(self):
        expected = '#!%s\n' % ei.nt_quote_arg(os.path.normpath(sys.executable))
        actual = ei.ScriptWriter.get_script_header('#!/usr/local/bin/python')
        assert actual == expected

    def test_get_script_header_args(self):
        expected = '#!%s -x\n' % ei.nt_quote_arg(os.path.normpath
            (sys.executable))
        actual = ei.ScriptWriter.get_script_header('#!/usr/bin/python -x')
        assert actual == expected

    def test_get_script_header_non_ascii_exe(self):
        actual = ei.ScriptWriter.get_script_header('#!/usr/bin/python',
            executable=self.non_ascii_exe)
        expected = '#!%s -x\n' % self.non_ascii_exe
        assert actual == expected

    def test_get_script_header_exe_with_spaces(self):
        actual = ei.ScriptWriter.get_script_header('#!/usr/bin/python',
            executable='"' + self.exe_with_spaces + '"')
        expected = '#!"%s"\n' % self.exe_with_spaces
        assert actual == expected


class TestCommandSpec:
    def test_custom_launch_command(self):
        """
        Show how a custom CommandSpec could be used to specify a #! executable
        which takes parameters.
        """
        cmd = ei.CommandSpec(['/usr/bin/env', 'python3'])
        assert cmd.as_header() == '#!/usr/bin/env python3\n'

    def test_from_param_for_CommandSpec_is_passthrough(self):
        """
        from_param should return an instance of a CommandSpec
        """
        cmd = ei.CommandSpec(['python'])
        cmd_new = ei.CommandSpec.from_param(cmd)
        assert cmd is cmd_new

    @mock.patch('sys.executable', TestScriptHeader.exe_with_spaces)
    @mock.patch.dict(os.environ)
    def test_from_environment_with_spaces_in_executable(self):
        os.environ.pop('__PYVENV_LAUNCHER__', None)
        cmd = ei.CommandSpec.from_environment()
        assert len(cmd) == 1
        assert cmd.as_header().startswith('#!"')

    def test_from_simple_string_uses_shlex(self):
        """
        In order to support `executable = /usr/bin/env my-python`, make sure
        from_param invokes shlex on that input.
        """
        cmd = ei.CommandSpec.from_param('/usr/bin/env my-python')
        assert len(cmd) == 2
        assert '"' not in cmd.as_header()


class TestWindowsScriptWriter:
    def test_header(self):
        hdr = ei.WindowsScriptWriter.get_script_header('')
        assert hdr.startswith('#!')
        assert hdr.endswith('\n')
        hdr = hdr.lstrip('#!')
        hdr = hdr.rstrip('\n')
        # header should not start with an escaped quote
        assert not hdr.startswith('\\"')
