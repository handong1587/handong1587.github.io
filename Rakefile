require 'rake'
require 'uglifier'
require 'fileutils'

task :default => :build

desc "Ensures all dependent JS libraries are installed and builds the gem."
task :build_gem => :build do
  exec("gem build jekyll-lunr-js-search.gemspec")
end

task :build => [
  :bower_update,
  :create_build_dir,
  :copy_jekyll_plugin,
  :concat_js,
  :minify_js]

task :bower_update do
  abort "Please ensure bower is installed: npm install -g bower" unless system('bower install')
end

task :create_build_dir do
  Dir.mkdir('build') unless Dir.exists?('build')
end

task :copy_jekyll_plugin do
  lunr_version = File.read("bower_components/lunr.js/VERSION").strip
  open("build/jekyll_lunr_js_search.rb", "w") do |concat|
    Dir.glob("lib/jekyll_lunr_js_search/*.rb") do |file|
      ruby = File.read(file).sub(/LUNR_VERSION = .*$/, "LUNR_VERSION = \"#{lunr_version}\"")
      concat.puts ruby
    end
  end
end

task :concat_js do
  files = [
    'bower_components/jquery/dist/jquery.js',
    'bower_components/mustache/mustache.js',
    'bower_components/date.format/date.format.js',
    'bower_components/uri.js/src/URI.js',
    'bower_components/lunr.js/lunr.min.js',
    'javascript/jquery.lunr.search.js'
  ]

  File.open('build/search.js', 'w') do |file|
    file.write(files.inject('') { |data, file|
      data << File.read(file)
    })
  end

  # Lunr is stored separately so we can use it for index generation
  FileUtils.cp('bower_components/lunr.js/lunr.min.js', 'build/lunr.min.js')
end

task :minify_js do
  minified, map = Uglifier.new.compile(File.read('build/search.js'))
  File.open('build/search.min.js', 'w') do |file|
    file.puts minified
  end
end

task :minify_js_map do
  minified, map = Uglifier.new.compile_with_map(File.read('build/search.js'))
  File.open('build/search.js.map', 'w') { |file| file.write(map) }
  File.open('build/search.min.js', 'w') do |file|
    file.puts minified
    file.write "//# sourceMappingURL=search.js.map"
  end
end