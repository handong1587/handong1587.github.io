(function($) {

  var debounce = function(fn) {
    var timeout;
    var slice = Array.prototype.slice;

    return function() {
      var args = slice.call(arguments),
          ctx = this;

      clearTimeout(timeout);

      timeout = setTimeout(function () {
        fn.apply(ctx, args);
      }, 100);
    };
  };

  // parse a date in yyyy-mm-dd format
  var parseDate = function(input) {
    var parts = input.match(/(\d+)/g);
    return new Date(parts[0], parts[1]-1, parts[2]); // months are 0-based
  };

  var LunrSearch = (function() {
    function LunrSearch(elem, options) {
      this.$elem = elem;
      this.$results = $(options.results);
      this.indexDataUrl = options.indexUrl;
      this.template = this.compileTemplate($(options.template));
      this.titleMsg = options.titleMsg;
      this.emptyMsg = options.emptyMsg;

      this.initialize();
    }

    LunrSearch.prototype.initialize = function() {
      var self = this;

      this.loadIndexData(function(data) {
        self.entries = $.map(data.docs, self.createEntry);
        self.index = lunr.Index.load(data.index);
        self.populateSearchFromQuery();
        self.bindKeypress();
      });
    };

    // compile search results template
    LunrSearch.prototype.compileTemplate = function($template) {
      var template = $template.text();
      Mustache.parse(template);
      return function (view, partials) {
        return Mustache.render(template, view, partials);
      };
    };

    // load the search index data
    LunrSearch.prototype.loadIndexData = function(callback) {
      $.getJSON(this.indexDataUrl, callback);
    };

    LunrSearch.prototype.createEntry = function(raw, index) {
      var entry = $.extend({
        id: index + 1
      }, raw);

      // include pub date for posts
      if (raw.date) {
        $.extend(entry, {
          date: parseDate(raw.date),
          pubdate: function() {
            // HTML5 pubdate
            return dateFormat(parseDate(raw.date), 'yyyy-mm-dd');
          },
          displaydate: function() {
            // only for posts (e.g. Oct 12, 2012)
            return dateFormat(parseDate(raw.date), 'mmm dd, yyyy');
          }
        });
      }

      return entry;
    };

    LunrSearch.prototype.bindKeypress = function() {
      var self = this;
      var oldValue = this.$elem.val();

      this.$elem.bind('keyup', debounce(function() {
        var newValue = self.$elem.val();
        if (newValue !== oldValue) {
          self.search(newValue);
        }

        oldValue = newValue;
      }));
    };

    LunrSearch.prototype.search = function(query) {
      var entries = this.entries;

      if (query.length < 3) {
        this.$results.hide().empty();
      } else {
        var results = $.map(this.index.search(query), function(result) {
          return $.grep(entries, function(entry) { return entry.id === parseInt(result.ref, 10); })[0];
        });

        this.displayResults(results);
      }
    };

    LunrSearch.prototype.displayResults = function(entries) {
      var $results = this.$results;

      $results.empty();

      if (entries.length === 0) {
        $results.append( this.emptyMsg );
      } else {
        if (this.titleMsg && 0 !== this.titleMsg.length) {
          $results.append( this.titleMsg );
        }
        $results.append(this.template({entries: entries}));
      }

      $results.show();
    };

    // Populate the search input with 'q' querystring parameter if set
    LunrSearch.prototype.populateSearchFromQuery = function() {
      var uri = new URI(window.location.search.toString());
      var queryString = uri.search(true);

      if (queryString.hasOwnProperty('q')) {
        this.$elem.val(queryString.q);
        this.search(queryString.q.toString());
      }
    };

    return LunrSearch;
  })();

  $.fn.lunrSearch = function(options) {
    // apply default options
    options = $.extend({}, $.fn.lunrSearch.defaults, options);

    // create search object
    new LunrSearch(this, options);

    return this;
  };

  $.fn.lunrSearch.defaults = {
    indexUrl  : '/javascript/index.json',           // url for the .json file containing search index data
    results   : '#search-results',          // selector for containing search results element
    template  : '#search-results-template', // selector for Mustache.js template
    titleMsg  : '<h1>Search results<h1>',   // message attached in front of results
    emptyMsg  : '<p>Nothing found.</p>'     // shown message if search returns no results
  };
})(jQuery);
