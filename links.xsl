<?xml version="1.0"?>
<!-- SPDX-License-Identifier: CC0-1.0 -->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:fr="http://www.jonmsterling.com/jms-005P.xml">

  <xsl:template match="text()[ancestor::fr:link[1]/@href]">
    <xsl:variable name="raw-href" select="ancestor::fr:link[1]/@href" />
    <xsl:variable name="check-href"
      select="translate(normalize-space($raw-href), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ&#9;&#10;&#13; ', 'abcdefghijklmnopqrstuvwxyz')" />
    <xsl:variable name="unsafe-scheme"
      select="starts-with($check-href, 'javascript:') or starts-with($check-href, 'vbscript:') or starts-with($check-href, 'data:') or starts-with($check-href, 'blob:') or starts-with($check-href, 'file:')" />
    <a>
      <xsl:attribute name="href">
        <xsl:choose>
          <xsl:when test="$unsafe-scheme">
            <xsl:text>#</xsl:text>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="$raw-href" />
          </xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>
      <xsl:if test="ancestor::fr:link[1]/@type = 'external'">
        <xsl:attribute name="rel">noopener noreferrer</xsl:attribute>
      </xsl:if>
      <xsl:choose>
        <xsl:when test="ancestor::fr:link[1]/@display-uri">
          <xsl:attribute name="title">
            <xsl:value-of select="ancestor::fr:link[1]/@title" />
            <xsl:text> [</xsl:text>
            <xsl:value-of select="ancestor::fr:link[1]/@display-uri" />
            <xsl:text>]</xsl:text>
          </xsl:attribute>
        </xsl:when>
        <xsl:otherwise>
          <xsl:attribute name="title">
            <xsl:value-of select="ancestor::fr:link[1]/@title" />
          </xsl:attribute>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:value-of select="."/>
    </a>
  </xsl:template>

  <xsl:template match="fr:link">
    <span class="link {@type}">
      <xsl:apply-templates  />
    </span>
  </xsl:template>

</xsl:stylesheet>
