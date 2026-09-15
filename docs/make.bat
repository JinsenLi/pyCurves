@ECHO OFF

if not defined SPHINXBUILD set SPHINXBUILD=sphinx-build

%SPHINXBUILD% -W -b html . _build/html
