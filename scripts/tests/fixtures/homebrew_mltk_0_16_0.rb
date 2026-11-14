class Mltk < Formula
  desc "Pytest for ML — unified testing across the entire ML lifecycle"
  homepage "https://github.com/Liorrr/mltk"
  url "https://files.pythonhosted.org/packages/80/86/248f1dd2b796920649d226c0843c9192923a8ef74086b119f2a0f1e4139f/mlspec-0.16.0.tar.gz"
  sha256 "2a927686cbd60583f62abb83265b13c067b59e2e7b45b7474e804b15a4c132d3"
  license "Elastic-2.0"

  depends_on "rust" => :build
  depends_on "python@3.13"

  on_linux do
    depends_on "zlib-ng-compat"
  end

  def install
    # Homebrew's venv.pip_install runs pip with --no-deps, and no runtime
    # dependency is declared as a resource, so it installed mlspec without
    # numpy, pandas, typer or rich. Let pip resolve them from PyPI instead.
    system formula_opt_bin("python@3.13")/"python3.13", "-m", "venv", libexec
    system libexec/"bin/python", "-m", "pip", "install", "#{buildpath}[cli]"
    bin.write_exec_script libexec/"bin/mltk"
  end

  def caveats
    <<~EOS
      mltk is currently published on PyPI as 'mlspec' while the 'mltk' PyPI name transfer is pending.
      For pip installs use: pip install mlspec
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/mltk version")
  end
end
