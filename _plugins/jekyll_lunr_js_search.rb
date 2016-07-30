require 'nokogiri'

module Jekyll
  module LunrJsSearch
    class PageRenderer
      def initialize(site)
        @site = site
      end
      
      # render item, but without using its layout
      def prepare(item)
        layout = item.data["layout"]
        begin
          item.data.delete("layout")

          if item.is_a?(Jekyll::Document)          
            output = Jekyll::Renderer.new(@site, item).run
          else
            item.render({}, @site.site_payload)
            output = item.output  
          end
        ensure
          # restore original layout
          item.data["layout"] = layout
        end
      
        output
      end

      # render the item, parse the output and get all text inside <p> elements
      def render(item)
        layoutless = item.dup

        Nokogiri::HTML(prepare(layoutless)).text
      end
    end
  end  
end
require "v8"
require "json"

class V8::Object
  def to_json
    @context['JSON']['stringify'].call(self)
  end

  def to_hash
    JSON.parse(to_json, :max_nesting => false)
  end
end
require 'fileutils'
require 'net/http'
require 'json'
require 'uri'
require 'v8'

module Jekyll
  module LunrJsSearch
    class Indexer < Jekyll::Generator
      def initialize(config = {})
        super(config)

        lunr_config = {
          'excludes' => [],
          'strip_index_html' => false,
          'min_length' => 3,
          'stopwords' => 'stopwords.txt',
          'fields' => {
            'title' => 10,
            'categories' => 20,
            'tags' => 20,
            'body' => 1
          },
          'js_dir' => 'js'
        }.merge!(config['lunr_search'] || {})

        @js_dir = lunr_config['js_dir']
        gem_lunr = File.join(File.dirname(__FILE__), "../../build/lunr.min.js")
        @lunr_path = File.exist?(gem_lunr) ? gem_lunr : File.join(@js_dir, File.basename(gem_lunr))
        raise "Could not find #{@lunr_path}" if !File.exist?(@lunr_path)

        ctx = V8::Context.new
        ctx.load(@lunr_path)
        ctx['indexer'] = proc do |this|
          this.ref('id')
          lunr_config['fields'].each_pair do |name, boost|
            this.field(name, { 'boost' => boost })
          end
        end
        @index = ctx.eval('lunr(indexer)')
        @lunr_version = ctx.eval('lunr.version')
        @docs = {}
        @excludes = lunr_config['excludes']

        # if web host supports index.html as default doc, then optionally exclude it from the url
        @strip_index_html = lunr_config['strip_index_html']

        # stop word exclusion configuration
        @min_length = lunr_config['min_length']
        @stopwords_file = lunr_config['stopwords']
      end

      # Index all pages except pages matching any value in config['lunr_excludes'] or with date['exclude_from_search']
      # The main content from each page is extracted and saved to disk as json
      def generate(site)
        Jekyll.logger.info "Lunr:", 'Creating search index...'

        @site = site
        # gather pages and posts
        items = pages_to_index(site)
        content_renderer = PageRenderer.new(site)
        index = []

        items.each_with_index do |item, i|
          entry = SearchEntry.create(item, content_renderer)

          entry.strip_index_suffix_from_url! if @strip_index_html
          entry.strip_stopwords!(stopwords, @min_length) if File.exists?(@stopwords_file)

          doc = {
            "id" => i,
            "title" => entry.title,
            "url" => entry.url,
            "date" => entry.date,
            "categories" => entry.categories,
            "tags" => entry.tags,
            "is_post" => entry.is_post,
            "body" => entry.body
          }

          @index.add(doc)
          doc.delete("body")
          @docs[i] = doc

          Jekyll.logger.debug "Lunr:", (entry.title ? "#{entry.title} (#{entry.url})" : entry.url)
        end

        FileUtils.mkdir_p(File.join(site.dest, @js_dir))
        filename = File.join(@js_dir, 'index.json')

        total = {
          "docs" => @docs,
          "index" => @index.to_hash
        }

        filepath = File.join(site.dest, filename)
        File.open(filepath, "w") { |f| f.write(JSON.dump(total)) }
        Jekyll.logger.info "Lunr:", "Index ready (lunr.js v#{@lunr_version})"
        added_files = [filename]

        site_js = File.join(site.dest, @js_dir)
        # If we're using the gem, add the lunr and search JS files to the _site
        if File.expand_path(site_js) != File.dirname(@lunr_path)
          extras = Dir.glob(File.join(File.dirname(@lunr_path), "*.min.js"))
          FileUtils.cp(extras, site_js)
          extras.map! { |min| File.join(@js_dir, File.basename(min)) }
          Jekyll.logger.debug "Lunr:", "Added JavaScript to #{@js_dir}"
          added_files.push(*extras)
        end

        # Keep the written files from being cleaned by Jekyll
        added_files.each do |filename|
          site.static_files << SearchIndexFile.new(site, site.dest, "/", filename)
        end
      end

      private

      # load the stopwords file
      def stopwords
        @stopwords ||= IO.readlines(@stopwords_file).map { |l| l.strip }
      end

      def output_ext(doc)
        if doc.is_a?(Jekyll::Document)
          Jekyll::Renderer.new(@site, doc).output_ext
        else
          doc.output_ext
        end
      end

      def pages_to_index(site)
        items = []

        # deep copy pages and documents (all collections, including posts)
        site.pages.each {|page| items << page.dup }
        site.documents.each {|document| items << document.dup }

        # only process files that will be converted to .html and only non excluded files
        items.select! {|i| i.respond_to?(:output_ext) && output_ext(i) == '.html' && ! @excludes.any? {|s| (i.url =~ Regexp.new(s)) != nil } }
        items.reject! {|i| i.data['exclude_from_search'] }

        items
      end
    end
  end
end
module Jekyll
  module LunrJsSearch
    VERSION = "3.1.0"
  end
end
require 'nokogiri'

module Jekyll
  module LunrJsSearch
    class SearchEntry
      def self.create(site, renderer)
        if site.is_a?(Jekyll::Page) or site.is_a?(Jekyll::Document)
          if defined?(site.date)
            date = site.date
          else
            date = nil
          end
          categories = site.data['categories']
          tags = site.data['tags']
          title, url = extract_title_and_url(site)
          is_post = site.is_a?(Jekyll::Document)
          body = renderer.render(site)

          SearchEntry.new(title, url, date, categories, tags, is_post, body, renderer)
        else
          raise 'Not supported'
        end
      end

      def self.extract_title_and_url(item)
        data = item.to_liquid
        [ data['title'], data['url'] ]
      end

      attr_reader :title, :url, :date, :categories, :tags, :is_post, :body, :collection

      def initialize(title, url, date, categories, tags, is_post, body, collection)
        @title, @url, @date, @categories, @tags, @is_post, @body, @collection = title, url, date, categories, tags, is_post, body, collection
      end

      def strip_index_suffix_from_url!
        @url.gsub!(/index\.html$/, '')
      end

      # remove anything that is in the stop words list from the text to be indexed
      def strip_stopwords!(stopwords, min_length)
        @body = @body.split.delete_if() do |x|
          t = x.downcase.gsub(/[^a-z]/, '')
          t.length < min_length || stopwords.include?(t)
        end.join(' ')
      end
    end
  end
end
module Jekyll
  module LunrJsSearch  
    class SearchIndexFile < Jekyll::StaticFile
      # Override write as the index.json index file has already been created 
      def write(dest)
        true
      end
    end
  end
end
