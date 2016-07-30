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
